"""Graph policy, scalar critic and exact-forward legal sparse branching."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from metro.models.multi_encoder import MultiObjectiveEncoder
from metro.models.stop_policy import StopPolicy
from metro.models.value import MetroValue
from .env import PreferenceState, normalize_weights
from .features import DynamicPotentials, legal_zscore, preference_graph


class LegalGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, target, legal):
        ctx.save_for_backward(legal)
        return target.clone()

    @staticmethod
    def backward(ctx, gradient):
        legal, = ctx.saved_tensors
        return gradient.masked_fill(~legal, 0.), None, None


def attach_legal_gradient(raw, target, mask):
    legal = mask > .5
    if raw.shape != target.shape or raw.shape != legal.shape:
        raise ValueError("Incompatible legal-gradient shapes.")
    if not torch.isfinite(raw).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite sparse logits.")
    target = torch.where(legal, target, torch.full_like(target, -1e9)).detach()
    return LegalGradient.apply(raw, target, legal)


@torch.no_grad()
def sparse_targets(raw, mask, potential, new_counts):
    legal = mask > .5
    count = potential.shape[1]
    target = torch.full_like(raw, -1e9)
    for row in range(raw.shape[0]):
        allowed = legal[row]
        if not allowed.any():
            raise ValueError("Empty action set.")
        best = int(raw[row].masked_fill(~allowed, -torch.inf).argmax())
        target[row, allowed] = -20.
        target[row, best] = 0.
        if float(new_counts[row]) < 8:
            target[row, allowed] = -25.
            target[row, best] = 0.
            continue
        candidates = torch.where(allowed[:count])[0]
        candidates = candidates[candidates != best]
        if len(candidates) == 0:
            continue
        raw_std = legal_zscore(raw[row:row + 1, :count], mask[row:row + 1])[0]
        q = .45 * raw_std + .55 * potential[row]
        first = candidates[raw[row, candidates].topk(min(4, len(candidates))).indices]
        second = candidates[potential[row, candidates].topk(min(3, len(candidates))).indices]
        pool = torch.unique(torch.cat([first, second]), sorted=True)
        alternatives = pool[q[pool].topk(min(2, len(pool))).indices]
        best_potential = potential[row, best] if best < count else potential.new_zeros(())
        best_q = q[best] if best < count else q.new_zeros(())
        chosen = int(alternatives[0])
        if float(new_counts[row]) < 24:
            gap = potential[row, chosen] - best_potential
            target[row, chosen] = (-2. + .45 * gap).clamp(-2.6, -.85) if gap > 0 else -4.
            if len(alternatives) > 1:
                target[row, int(alternatives[1])] = -4.5
        else:
            target[row, chosen] = (-1.15 + .50 * (q[chosen] - best_q)).clamp(-2., -.05)
            if len(alternatives) > 1:
                other = int(alternatives[1])
                target[row, other] = (-1.85 + .45 * (q[other] - best_q)).clamp(-2.8, -.35)
    return target


def tensor_state(state, device):
    parts = [torch.as_tensor(np.array(item, copy=True), device=device) for item in state]
    return PreferenceState(parts, state.preference_w, state.decoder_mode)


class ActorCritic(nn.Module):
    def __init__(self, env, config, target):
        super().__init__()
        self.target = normalize_weights(target)
        winners = np.flatnonzero(self.target == self.target.max())
        primary = int(winners[0]) if len(winners) == 1 else -1
        self.profile = "commute" if primary == 1 else ("equity" if primary == 2 else "guided")
        if self.profile == "commute":
            env.core.edge_index_od = preference_graph(env, self.target)
        self.node_width = 40 if self.profile == "equity" else 44
        agent = SimpleNamespace(env=env, node_dim=self.node_width,
            numerical_feature_size=env.get_numerical_feature_size(), dtype=torch.float32)
        self.encoder = MultiObjectiveEncoder(config["state_encoder_specs"], agent)
        self.policy = StopPolicy(config["policy_specs"], agent, self.encoder)
        self.critic = MetroValue(config["value_specs"], agent, self.encoder)
        if self.profile != "equity":
            self.prior_strength = nn.Parameter(torch.tensor(3. / 4.))
            nn.init.constant_(self.policy.stop_head[-1].bias, -1.5)
        else:
            self.register_parameter("prior_strength", None)
        self.potentials = DynamicPotentials(env, self.target)

    def prepare(self, states):
        device = next(self.parameters()).device
        tensors = [tensor_state(state, device) for state in states]
        nodes = torch.stack([state[1] for state in tensors]).float()
        mask = torch.stack([state[-2] for state in tensors])
        weights = torch.tensor(np.stack([state.preference_w for state in states]), device=device)
        need = self.profile != "equity" or any(state.decoder_mode for state in states)
        auxiliary = guide = None
        if need:
            dynamic, guide, auxiliary = self.potentials(nodes, weights, mask,
                                                       compact=self.profile == "commute")
            if self.node_width > 40:
                for i, state in enumerate(tensors):
                    state[1] = torch.cat([nodes[i], dynamic[i]], dim=-1)
        return tensors, guide, auxiliary

    def forward(self, states, critic_only=False):
        tensors, guide, auxiliary = self.prepare(states)
        if critic_only:
            with torch.no_grad():
                _, hidden, _, _ = self.encoder(tensors)
            return self.critic.value_head(hidden.detach()).flatten()
        policy_embed, hidden, mask, _ = self.encoder(tensors)
        raw = torch.cat([self.policy.policy_road_head(policy_embed),
                         self.policy.stop_head(policy_embed.mean(1))], dim=1)
        if self.prior_strength is not None:
            node = raw[:, :-1] + self.prior_strength.clamp(.1, 2.) * guide
            raw = torch.cat([node, raw[:, -1:]], dim=1)
        if any(state.decoder_mode for state in states):
            tail = self.potentials.tail(guide, auxiliary, mask)
            counts = auxiliary["built"].sum(1) - self.potentials.initial_station_count
            target = sparse_targets(raw.detach(), mask, tail, counts)
            enabled = torch.tensor([bool(state.decoder_mode) for state in states],
                                   device=raw.device)[:, None]
            safe_raw = raw.masked_fill(~mask.bool(), -1e9)
            target = torch.where(enabled, target, safe_raw.detach())
            logits = attach_legal_gradient(raw, target, mask)
        else:
            logits = raw.masked_fill(~mask.bool(), -1e9)
        distribution = torch.distributions.Categorical(logits=logits)
        value = self.critic.value_head(hidden).flatten()
        return distribution, value
