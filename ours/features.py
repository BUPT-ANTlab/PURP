"""Fixed-scale multi-objective potentials on the candidate network."""
import numpy as np
import torch
from torch import nn


def legal_zscore(values, mask, limit=2.5):
    legal = mask[:, :values.shape[1]] > .5
    count = legal.sum(1, keepdim=True)
    clean = values.masked_fill(~legal, 0.)
    if not torch.isfinite(clean).all():
        raise FloatingPointError("Nonfinite legal potential.")
    mean = clean.sum(1, keepdim=True) / count.clamp_min(1)
    delta = (clean - mean).masked_fill(~legal, 0.)
    variance = delta.square().sum(1, keepdim=True) / count.clamp_min(1)
    std = torch.sqrt(variance.clamp_min(0.) + 1e-6)
    result = torch.where(count > 1, delta / std, torch.zeros_like(delta))
    return result.masked_fill(~legal, 0.).clamp(-limit, limit)


def preference_graph(env, target):
    scenario, arrays = env.scenario, env.bundle["arrays"]
    general = scenario.flow + scenario.flow.T
    commute = arrays["M_JH"] + arrays["M_JH"].T
    fused = (target[0] * general / max(float(general.max()), 1e-12)
             + target[1] * commute / max(float(commute.max()), 1e-12))
    u, v = np.triu_indices(len(scenario.nodes), k=1)
    order = np.argsort(-fused[u, v], kind="stable")[:len(env.core.edge_index_od)]
    return np.stack([u[order], v[order]], axis=1).astype(np.int64)


class DynamicPotentials(nn.Module):
    def __init__(self, env, target):
        super().__init__()
        operators = env.bundle["operators"]
        for name in ("P_short", "P_mid", "P_long", "K_tau", "M_JH_sym", "jh_row_sum", "v_prob"):
            self.register_buffer(name, torch.tensor(operators[name], dtype=torch.float32), persistent=False)
        self.register_buffer("K7", torch.tensor(np.exp(-env.scenario.direct / 7.),
                                              dtype=torch.float32), persistent=False)
        self.register_buffer("direct", torch.tensor(env.scenario.direct, dtype=torch.float32), persistent=False)
        self.initial_station_count = int(env.scenario.initial_active.sum())
        self.station_cost = float(env.scenario.cost[1])
        self.distance_cost = float(env.scenario.cost[0])
        nodes = torch.tensor(env.observation()[1])[None]
        built = (nodes[:, :, 30] > .5).float()
        acc = self.accessibility(built)
        general = nodes[:, :, 26]
        commute = .65 * (built @ self.M_JH_sym) + .35 * self.jh_row_sum
        fair = self.fair_gain(acc)
        pull = (self.v_prob * (1. - acc)) @ self.K7
        scales = torch.stack([general.max(), commute.max(), fair.max(), pull.max()]) + 1e-6
        scales[0] = float(np.asarray(operators["scale_values"])[3])
        self.register_buffer("source_scales", scales)
        raw = torch.stack([general / scales[0], commute / scales[1], fair / scales[2]], dim=-1)
        nav = self.diffuse((1. - built)[:, :, None] * raw)
        self.register_buffer("nav_scales", nav.amax(dim=(0, 1)) + 1e-6)
        mixed = (raw * torch.as_tensor(target, dtype=torch.float32)[None, None]).sum(-1)
        mixed_nav = torch.stack([(mixed * (1. - built)) @ p
                                  for p in (self.P_short, self.P_mid, self.P_long)], dim=-1)
        self.register_buffer("compact_scales", mixed_nav.amax(dim=(0, 1)) + 1e-6)

    def accessibility(self, built):
        return torch.cat([(self.K_tau[i:i + 128][None] * built[:, None]).amax(-1)
                          for i in range(0, built.shape[1], 128)], dim=1)

    def fair_gain(self, acc):
        return torch.cat([torch.relu(self.K_tau[i:i + 64][None] - acc[:, None]) @ self.v_prob
                          for i in range(0, acc.shape[1], 64)], dim=1)

    def diffuse(self, sources):
        return torch.stack([(sources.transpose(1, 2) @ p).transpose(1, 2)
                            for p in (self.P_short, self.P_mid, self.P_long)], dim=-1)

    @torch.no_grad()
    def forward(self, nodes, weights, mask, compact=False):
        built = (nodes[:, :, 30] > .5).float()
        unbuilt = 1. - built
        acc = self.accessibility(built)
        c = self.source_scales
        general = nodes[:, :, 26] / c[0]
        commute = (.65 * (built @ self.M_JH_sym) + .35 * self.jh_row_sum) / c[1]
        fair = self.fair_gain(acc) / c[2]
        pull = ((self.v_prob * (1. - acc)) @ self.K7) / c[3]
        raw = torch.stack([general, commute, fair], dim=-1)
        nav = self.diffuse(unbuilt[:, :, None] * raw) / self.nav_scales
        if compact:
            mixed = (raw * weights[:, None]).sum(-1)
            fields = torch.stack([(unbuilt * mixed) @ p
                                 for p in (self.P_short, self.P_mid, self.P_long)], dim=-1)
            fields = fields / self.compact_scales
            dynamic = torch.cat([mixed[:, :, None], fields], dim=-1)
            guide = legal_zscore(.45 * mixed + .30 * fields[:, :, 1]
                                 + .25 * fields[:, :, 2], mask)
        else:
            scores = torch.stack([
                unbuilt * (.50 * general + .30 * nav[:, :, 0, 1] + .20 * nav[:, :, 0, 2]),
                unbuilt * (.45 * commute + .30 * nav[:, :, 1, 1] + .25 * nav[:, :, 1, 2]),
                unbuilt * (.40 * fair + .35 * pull + .25 * nav[:, :, 2, 2])], dim=-1)
            channels = torch.stack([legal_zscore(scores[:, :, k], mask) for k in range(3)], dim=-1)
            guide = legal_zscore((channels * weights[:, None]).sum(-1), mask)
            dynamic = torch.cat([channels, guide[:, :, None]], dim=-1)
        auxiliary = {"built": built, "acc": acc, "fair": fair, "pull": pull}
        return dynamic, guide, auxiliary

    @torch.no_grad()
    def tail(self, guide, auxiliary, mask):
        built, acc = auxiliary["built"], auxiliary["acc"]
        nearest = self.direct[None].masked_fill(~built[:, None].bool(), torch.inf).amin(-1)
        cost = (self.station_cost + self.distance_cost * nearest).clamp_min(1e-6)
        efficiency = legal_zscore((auxiliary["fair"] + .6 * auxiliary["pull"]) / cost, mask)
        cluster = legal_zscore((self.v_prob * (1. - acc)) @ self.P_long, mask)
        return legal_zscore(.45 * guide + .35 * efficiency + .20 * cluster, mask)
