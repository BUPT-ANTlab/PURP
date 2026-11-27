"""Local multi-objective metro environment and independent route evaluation."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
from metro.objectives import load_scenario, ObjectiveSpec
from metro.multi_env import MultiObjectiveEnv

ROOT = Path(__file__).resolve().parents[1]


def normalize_weights(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Supply three finite nonnegative preference coefficients.")
    largest = values.max()
    if largest <= 0:
        raise ValueError("At least one preference coefficient must be positive.")
    values = values / largest
    return values / values.sum()


def score_flow(distance, active, direct, flow):
    ids = np.flatnonzero(active)
    block = np.ix_(ids, ids)
    path, geo = distance[block], direct[block]
    ratio = np.divide(geo, path, out=np.zeros_like(geo), where=path > 0)
    credit = np.where(geo <= 3., 1., np.clip(ratio, .2, .8))
    credit[~np.isfinite(path)] = 0.
    return float(np.sum(flow[block] * credit))


@lru_cache(maxsize=4)
def load_bundle(city="beijing", initial_lines=4):
    folder = ROOT / "data" / city / f"{initial_lines}lines" / "derived"
    names = ("dataset.npz", "operators.npz", "commute_features.npz",
             "equity_features.npz", "reference_scales.json")
    if not all((folder / name).is_file() for name in names):
        raise ValueError("Static objective assets are required for the selected network.")
    tables = {}
    for name in names[:-1]:
        with np.load(folder / name, allow_pickle=False) as archive:
            tables[name] = {key: archive[key].copy() for key in archive.files}
    refs = json.loads((folder / names[-1]).read_text())
    scales = np.asarray(refs["scales"], dtype=np.float64)
    if refs["names"] != ["G", "J", "F"] or scales.shape != (3,) or not (scales > 0).all():
        raise ValueError("Invalid objective reference scales.")
    scenario = load_scenario(city, initial_lines, data_root=str(ROOT / "data"))
    arrays = tables["dataset.npz"]
    count = len(scenario.nodes)
    np.testing.assert_array_equal(arrays["node_ids"], [str(n) for n in scenario.nodes])
    np.testing.assert_allclose(arrays["positions_lat_lon"], scenario.positions, rtol=0, atol=1e-10)
    np.testing.assert_allclose(arrays["a"], scenario.initial_nearest, rtol=0, atol=1e-10)
    if arrays["base_static_features"].shape != (count, 26):
        raise ValueError("Invalid static physical feature dimensions.")
    if arrays["extra_features"].shape != (count, 4) or arrays["j_flow"].shape != (count, count):
        raise ValueError("Invalid objective feature dimensions.")
    for table in tables.values():
        for value in table.values():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError("Nonfinite static asset.")
            value.setflags(write=False)
    v = arrays["v"]
    if v.shape != (count,) or (v < 0).any() or not np.isclose(v.sum(), 1.):
        raise ValueError("Invalid fairness probability measure.")
    operators = tables["operators.npz"]
    for name in ("P_short", "P_mid", "P_long", "K_tau", "M_JH_sym"):
        if operators[name].shape != (count, count):
            raise ValueError("Invalid network operator dimensions.")
    np.testing.assert_allclose(operators["v_prob"], v, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(operators["K_tau"], np.exp(-scenario.direct / 3.), atol=1e-6)
    np.testing.assert_allclose(operators["M_JH_sym"], arrays["M_JH"] + arrays["M_JH"].T,
                               rtol=1e-6, atol=1e-6)
    extra = arrays["extra_features"]
    np.testing.assert_allclose(extra[:, 2], v / v.max(), atol=1e-6)
    np.testing.assert_allclose(extra[:, 3], -np.expm1(-arrays["a"] / 3.), atol=1e-6)
    if extra.min() < 0 or extra.max() > 1 + 1e-6:
        raise ValueError("Static observation features must lie in the unit interval.")
    fingerprint = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}
    scales.setflags(write=False)
    return {"scenario": scenario, "arrays": arrays, "operators": tables["operators.npz"],
            "commute": tables["commute_features.npz"], "equity": tables["equity_features.npz"],
            "refs": scales, "fingerprint": fingerprint}


class PreferenceState(list):
    """Six physical observation fields with transition-local decoding context."""
    def __init__(self, parts, preference_w, decoder_mode=0):
        if len(parts) != 6:
            raise ValueError("Exactly six physical observation fields required.")
        super().__init__(parts)
        self.preference_w = normalize_weights(preference_w).astype(np.float32)
        self.preference_w.setflags(write=False)
        self.decoder_mode = int(decoder_mode)

    def clone(self):
        return PreferenceState([value.clone() if hasattr(value, "clone") else value.copy()
                                for value in self], self.preference_w, self.decoder_mode)


class GJFEnv:
    def __init__(self, weights, city="beijing", initial_lines=4, budget=40.,
                 max_lines=3, min_stations=10, bundle=None):
        self.bundle = bundle or load_bundle(city, initial_lines)
        self.scenario = self.bundle["scenario"]
        self.w = normalize_weights(weights)
        self.budget, self.max_lines, self.min_stations = budget, max_lines, min_stations
        self.inner = MultiObjectiveEnv(scenario=self.scenario, budget=budget,
            weights=self.w, spec=ObjectiveSpec(self.bundle["refs"][0]),
            pre_line=initial_lines, max_line=max_lines, min_num=min_stations, expand=False)
        self.core = self.inner.core
        table = self.bundle["arrays"]["base_static_features"]
        self.core.graph_node_feature = {n: table[i].tolist() for i, n in enumerate(self.core.node_list)}
        self.refs = self.bundle["refs"]
        self.stop_action = self.inner.stop_action
        self.j_initial = score_flow(self.scenario.initial_distance, self.scenario.initial_active,
                                    self.scenario.direct, self.bundle["arrays"]["j_flow"])
        self.initial_service = np.exp(-self.bundle["arrays"]["a"] / 3.)
        self.decoder_mode = 0
        self.reset()

    @property
    def done(self):
        return self.inner.done

    @property
    def feasible(self):
        return self.inner.feasible

    def get_node_dim(self):
        return 40

    def get_numerical_feature_size(self):
        return self.inner.get_numerical_feature_size()

    def current(self):
        arrays = self.bundle["arrays"]
        return np.array([self.core.total_od,
            score_flow(self.core.distance, self.core.active, self.scenario.direct,
                       arrays["j_flow"]) - self.j_initial,
            100. * np.dot(arrays["v"], np.exp(-self.core.nearest / 3.) - self.initial_service)])

    def observation(self):
        parts = list(self.inner.observation())
        parts[1] = np.concatenate([parts[1],
            self.bundle["arrays"]["extra_features"].astype(np.float32)], axis=1)
        return PreferenceState(parts, self.w, self.decoder_mode)

    def reset(self, weights=None, decoder_mode=None):
        if weights is not None:
            self.w = normalize_weights(weights)
        if decoder_mode is not None:
            self.decoder_mode = int(decoder_mode)
        self.inner.weights = tuple(self.w)
        self.inner.reset()
        self.raw = self.current()
        np.testing.assert_allclose(self.raw, 0., rtol=0, atol=1e-10)
        self.return_vector = np.zeros(3)
        return self.observation()

    def step(self, action):
        before = self.raw.copy()
        _, _, done, physical_info = self.inner.step(action)
        self.raw = self.current()
        vector = (self.raw - before) / self.refs
        if (vector < -1e-8).any():
            raise AssertionError("A physical gain decreased.")
        self.return_vector += vector
        if done:
            if not self.feasible or self.core.incomplete:
                raise RuntimeError("Incomplete or infeasible episode.")
            np.testing.assert_allclose(self.return_vector, self.raw / self.refs, atol=1e-8)
        return self.observation(), vector, done, {
            "raw_GJF": self.raw.copy(), "reward_scalar": float(self.w @ vector),
            "feasible": self.feasible, "termination_reason": physical_info["termination_reason"]}

    def independent(self):
        lines = self.core.lines()
        physical = self.scenario.evaluate(lines, self.budget, self.max_lines, self.min_stations)
        distance, active = self.scenario.distances(lines)
        nearest = self.scenario.nearest_distances(active)
        vector = np.array([physical["od_gain"],
            score_flow(distance, active, self.scenario.direct, self.bundle["arrays"]["j_flow"])
            - self.j_initial,
            100. * np.dot(self.bundle["arrays"]["v"],
                         np.exp(-nearest / 3.) - self.initial_service)])
        return vector, physical

    def plan(self):
        if not self.done or not self.feasible or self.core.incomplete:
            raise ValueError("Only completed feasible networks may be exported.")
        vector, physical = self.independent()
        np.testing.assert_allclose(vector, self.raw, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(physical["cost"], self.core.total_cost, atol=1e-8)
        if not physical["feasible"]:
            raise AssertionError("Independent physical legality check failed.")
        return {"raw_GJF": vector.tolist(), "normalized_GJF": (vector / self.refs).tolist(),
            "weights": self.w.tolist(), "utility": float(self.w @ (vector / self.refs)),
            "cost": physical["cost"], "remaining_budget": self.budget - physical["cost"],
            "budget": self.budget, "feasible": True,
            "lines": self.scenario.encode_lines(self.core.lines()),
            "actions": list(self.inner.actions)}
