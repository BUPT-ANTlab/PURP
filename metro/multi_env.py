"""Corrected metro dynamics, reusing the author's graph features and line geometry."""
import copy
import numpy as np
from metro.envs.metro import Metro, MLine
from metro.completion import CompletionGuard, COMPLETION_RULE
from metro.objectives import ATOL, RTOL, VERSION, ObjectiveSpec, memberships, load_scenario, validate_weights


class MetroCore(Metro):
    def __init__(self, scenario, budget=40, max_line=3, min_num=10, expand=False,
                 allow_new_lines=True):
        self.scenario = scenario
        self.allow_new_lines = allow_new_lines
        self.actions = {}
        self._memberships = {}
        self.completion_guard = CompletionGuard(self)
        data = [scenario.graph, scenario.regions, scenario.od, scenario.line_info]
        super().__init__(data, budget, max_line, min_num, expand, scenario.cost)
        self._cal_node_endpoint()
        self.edge_index_dis = np.asarray(self.edge_index_dis, dtype=np.int64).reshape(-1, 2)
        self.edge_index_od = np.asarray(self.edge_index_od, dtype=np.int64).reshape(-1, 2)
        self.max_steps = self.G.number_of_edges() - len(scenario.initial_edges) + 1

    def init_min_dis(self):
        self.distance = self.scenario.initial_distance.copy()
        self.active = self.scenario.initial_active.copy()
        self.score = self.scenario.initial_score
        self.nearest = self.scenario.initial_nearest.copy()
        self.access_gain = 0.
        self.total_cost = self.total_od = self.reward = 0.
        self._memberships = memberships(self.lines())
        for u, v, edge in self.G.edges(data=True):
            edge["cost"] = edge["length"] * self.cost_per_kmeter + self.cost_per_station

    def lines(self):
        return {line.name: list(line.stations) for line in self.metro_lines}

    @property
    def incomplete(self):
        return any(line.name not in self.scenario.initial_lines
                   and len(set(line.stations)) < self.min_num for line in self.metro_lines)

    def _get_node_loc(self, node):
        low = self.scenario.positions.min(axis=0)
        span = np.maximum(self.scenario.positions.max(axis=0) - low, 1e-12)
        return ((self.node_pos[node] - low) / span).tolist()

    def _cal_node_endpoint(self):
        self.node_is_endpoint = {node: [0, int(node in self.pre_build_node),
                                       int(node in self._memberships)] for node in self.node_list}
        for line in self.metro_lines:
            start, end = line.get_start_and_end_id()
            if start != end:
                self.node_is_endpoint[start][0] = 1
                self.node_is_endpoint[end][0] = 1

    def reset(self, eval=False):
        self.completion_guard.clear()
        self.build_new_line, self.building_line = True, None
        self.done = self.build_line_num = 0
        self.build_node, self.build_edge, self.edge_build_idx = [], [], []
        self.station = {node: 0 for node in self.node_list}
        self.seq = {node: [0, 0, 0, 0] for node in self.node_list}
        self.build_node_od = {}
        MLine.tot_seq = 0
        self.init_build_line()
        self.init_min_dis()
        self._refresh_memberships()
        self._cal_mask()
        self._cal_node_endpoint()

    def _refresh_memberships(self):
        self._memberships = memberships(self.lines())
        self.station = {node: len(self._memberships.get(node, ())) for node in self.node_list}
        self.seq = {node: [0, 0, 0, 0] for node in self.node_list}
        for line in self.metro_lines:
            seen = set()
            for position, node in enumerate(line.stations, 1):
                if node in seen:
                    continue
                seen.add(node)
                if self.seq[node][:2] == [0, 0]:
                    self.seq[node][:2] = [line.seq, position]
                elif self.seq[node][2:] == [0, 0]:
                    self.seq[node][2:] = [line.seq, position]

    def incremental_cost(self, u, v, name):
        cost = self.G[u][v]["length"] * self.cost_per_kmeter
        for node in (u, v):
            before = self._memberships.get(node, set())
            if not before:
                cost += self.cost_per_station
            if len(before) < 2 and len(before | {name}) >= 2:
                cost += self.cost_per_trans_station
        return float(cost)

    @staticmethod
    def _cos(a, b):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (norm + 1e-6)) if norm > 1e-12 else 0.

    def _cal_mask(self):
        self.actions, self.build_old_line_dict, self.build_new_line_dict = {}, {}, {}
        if self.incomplete:
            lines = [self.building_line]
        elif self.build_line_num >= self.max_line and self.building_line is not None:
            lines = self.metro_lines
        elif not self.build_new_line or not self.expand:
            lines = [self.building_line] if self.building_line is not None else self.metro_lines
        else:
            lines = self.metro_lines
        used = set(self.edge_build_idx)
        for line in lines:
            if line is None:
                continue
            for end, direction, sign in ((line.start_id, line.vstart, 1),
                                         (line.end_id, line.vend, -1)):
                for node in self.G.neighbors(end):
                    edge = self.G[end][node]
                    vector = self.node_pos[node] - self.node_pos[end]
                    if (edge["idx"] in used or node in line.stations
                            or self._cos(direction, vector) <= .3
                            or sign * self._cos(line.vline, vector) <= .3):
                        continue
                    cost = self.incremental_cost(end, node, line.name)
                    if self.total_cost + cost <= self.budget + ATOL:
                        self.build_old_line_dict[node] = [end, line]
                        self.actions[self.node_list_idx[node]] = (end, node, line.name, False, cost)
        may_start = (self.allow_new_lines and self.build_new_line and not self.incomplete
                     and self.build_line_num < self.max_line)
        if may_start:
            name = f"new_{len(self.metro_lines)+1}"
            for built in self.pre_build_node:
                for node in self.G.neighbors(built):
                    edge = self.G[built][node]
                    cost = self.incremental_cost(built, node, name)
                    if (edge["idx"] not in used and node not in self.build_old_line_dict
                            and self.total_cost + cost <= self.budget + ATOL):
                        previous = self.build_new_line_dict.get(node)
                        if previous is None or edge["length"] < self.G[previous][node]["length"]:
                            self.build_new_line_dict[node] = built
                            self.actions[self.node_list_idx[node]] = (built, node, name, True, cost)
            # Preserve the original new-line separation restriction.
            for node in self.build_node:
                if node not in self.pre_build_node:
                    for neighbor in self.G.neighbors(node):
                        self.actions.pop(self.node_list_idx[neighbor], None)
        for action, resolved in list(self.actions.items()):
            if not self.completion_guard.allows(resolved):
                self.actions.pop(action)
                node = self.node_list[action]
                self.build_old_line_dict.pop(node, None)
                self.build_new_line_dict.pop(node, None)
        self.mask = np.zeros(self.node_num, dtype=np.int64)
        self.mask[list(self.actions)] = 1
        self.done = int(not self.actions)

    def fake_cost(self, action):
        if int(action) not in self.actions:
            raise ValueError(f"Infeasible construction action {action}")
        return self.actions[int(action)][4]

    def add_station_from_action(self, action):
        action = int(action)
        if action not in self.actions:
            raise ValueError(f"Infeasible construction action {action}")
        self._pending = self.actions[action]
        # Reuse the original sequence bookkeeping and line extension mechanism.
        super().add_station_from_action(action)
        self._refresh_memberships()
        self._cal_node_endpoint()
        self._cal_mask()

    def update_dis_cost_reward(self, old_station, new_station, is_transfer, is_new_line):
        before = self.score
        u, v = self.node_list_idx[old_station], self.node_list_idx[new_station]
        new_ids = [i for i in (u, v) if not self.active[i]]
        self.active[u] = self.active[v] = True
        if new_ids:
            self.nearest = np.minimum(
                self.nearest,
                self.scenario.direct[:, new_ids].min(axis=1),
            )
        self.access_gain = (
            self.scenario.initial_access_distance - float(self.nearest.mean())
        ) / self.scenario.initial_access_distance
        ids = np.flatnonzero(self.active)
        block = np.ix_(ids, ids)
        length = self.G[old_station][new_station]["length"]
        # All operands are copies of the pre-update distances.
        du, dv = self.distance[ids, u].copy(), self.distance[ids, v].copy()
        old = self.distance[block]
        self.distance[block] = np.minimum(old, np.minimum(
            du[:, None] + length + dv[None, :],
            dv[:, None] + length + du[None, :]))
        self.score = self.scenario.score(self.distance, self.active)
        self.reward = self.score - before
        self.total_od = self.score - self.scenario.initial_score
        self.total_cost += self._pending[4]
        if new_station not in self.build_node:
            self.build_node.append(new_station)
        self.edge_build_idx.append(self.G[old_station][new_station]["idx"])

    def get_reward(self):
        return self.reward

    def get_numerical_dim(self):
        return 10

    def get_node_dim(self):
        return super().get_node_dim() + 1

    def _get_node_feature(self, node):
        access = float(
            self.nearest[self.node_list_idx[node]]
            / self.scenario.initial_access_distance)
        return super()._get_node_feature(node) + [access]

    def _get_numerical(self):
        progress = (len(set(self.building_line.stations)) / self.min_num
                    if self.building_line is not None else 0.)
        return super()._get_numerical() + [min(progress, 1.), float(self.incomplete), self.access_gain]

    def get_obs(self):
        numerical = np.asarray(self._get_numerical(), dtype=np.float32)
        features = np.asarray([self._get_node_feature(n) for n in self.node_list], dtype=np.float32)
        return numerical, features, self.edge_index_dis, self.edge_index_od, self.mask.copy()


class MultiObjectiveEnv:
    def __init__(self, city="changsha", budget=40, weights=None, spec=None, scenario=None,
                 pre_line=4, max_line=3, min_num=10, expand=False, allow_new_lines=True):
        if not np.isfinite(budget) or not 0 < budget <= 60:
            raise ValueError("Require 0 < budget <= 60")
        weights = validate_weights(weights)
        if max_line < 0 or min_num < 2:
            raise ValueError("Invalid line count/minimum station count")
        self.scenario = scenario or load_scenario(city, pre_line)
        self.spec = spec or ObjectiveSpec(1.)
        self.weights, self.budget = weights, float(budget)
        self.max_line, self.min_num, self.expand = max_line, min_num, expand
        self.core = MetroCore(self.scenario, budget, max_line, min_num, expand, allow_new_lines)
        self.stop_action = self.core.node_num
        self.reset()

    def get_node_dim(self):
        return self.core.get_node_dim()

    def get_numerical_feature_size(self):
        return self.core.get_numerical_dim()

    def observation(self):
        values = list(self.core.get_obs())
        values[-1] = np.append(values[-1], int(not self.core.incomplete)).astype(np.int64)
        if self.done:
            values[-1][:] = 0
            values[-1][-1] = 1
        return values + [np.array([not self.done, self.done], dtype=np.float32)]

    def reset(self, eval=False):
        self.core.reset(eval)
        self.done, self.feasible = False, True
        self.actions, self.steps = [], 0
        self.episode_return = 0.
        self.reason = None
        return self.observation()

    def utility(self):
        return self.spec.utility(self.core.total_od, self.core.access_gain, self.core.total_cost, self.weights)

    def step(self, action, logger=None):
        if self.done:
            raise RuntimeError("Action after terminal state")
        arr = np.asarray(action)
        if arr.size != 1 or not np.isfinite(arr).all() or float(arr.item()) != int(arr.item()):
            raise ValueError("Action must be a finite scalar integer")
        action = int(arr.item())
        if not 0 <= action <= self.stop_action:
            raise ValueError("Action outside action space")
        allowed = (not self.core.incomplete) if action == self.stop_action else bool(self.core.mask[action])
        if not allowed:
            raise ValueError("Action is masked")
        before_gain, before_cost = self.core.total_od, self.core.total_cost
        before_access = self.core.access_gain
        self.steps += 1
        self.actions.append(action)
        if action == self.stop_action:
            self.done, self.reason = True, "stop"
        else:
            self.core.add_station_from_action(action)
            if not self.core.actions:
                self.done = True
                self.feasible = not self.core.incomplete
                self.reason = "no_legal_actions" if self.feasible else "incomplete_new_line"
        if self.steps >= self.core.max_steps and not self.done:
            self.done, self.feasible, self.reason = True, False, "step_limit"
        vector = [
            float(self.core.total_od - before_gain),
            float(self.core.access_gain - before_access),
            float(-(self.core.total_cost - before_cost)),
        ]
        reward = float(np.dot(
            self.weights,
            np.asarray(vector) / self.spec.metadata()["scales"]))
        penalty = 0.
        if self.done and not self.feasible:
            penalty = -2. - self.utility()
            reward += penalty
        self.episode_return += reward
        info = {"reward": float(reward), "objective_reward": vector, "training_penalty": penalty,
                "od_score": self.core.score, "od_gain": self.core.total_od,
                "access_gain": self.core.access_gain,
                "access_distance_km": float(self.core.nearest.mean()),
                "cost": self.core.total_cost, "utility": self.utility(),
                "episode_return": self.episode_return, "feasible": self.feasible,
                "termination_reason": self.reason, "steps": self.steps,
                "completion_guard": dict(self.core.completion_guard.stats)}
        return self.observation(), float(reward), self.done, info

    def snapshot(self, provenance=None):
        result = self.scenario.evaluate(self.core.lines(), self.budget, self.max_line, self.min_num)
        if not np.isclose(result["od_gain"], self.core.total_od, rtol=RTOL, atol=ATOL):
            raise AssertionError("Independent OD evaluation disagrees with environment")
        if not np.isclose(result["cost"], self.core.total_cost, rtol=RTOL, atol=ATOL):
            raise AssertionError("Independent asset cost disagrees with environment")
        if not np.isclose(
                result["access_gain"], self.core.access_gain,
                rtol=RTOL, atol=ATOL):
            raise AssertionError(
                "Independent access evaluation disagrees with environment")
        if not np.isclose(
                result["access_distance_km"], self.core.nearest.mean(),
                rtol=RTOL, atol=ATOL):
            raise AssertionError(
                "Independent access distance disagrees with environment")
        if bool(result["feasible"]) != self.feasible:
            raise AssertionError("Independent feasibility disagrees with environment")
        return {**result, "city": self.scenario.city, "budget": self.budget,
                "scenario_hash": self.scenario.hash, "objectives": self.spec.metadata(),
                "weights": list(self.weights), "utility": self.utility(),
                "stations": {
                    str(n): self.scenario.positions[i].tolist()
                    for i, n in enumerate(self.scenario.nodes)
                    if self.core.active[i]
                },
                "episode_return": self.episode_return, "termination_reason": self.reason,
                "constraints": {"max_new_lines": self.max_line, "min_stations": self.min_num,
                                "expand": self.expand, "completion_guard": dict(COMPLETION_RULE)},
                "lines": self.scenario.encode_lines(self.core.lines()), "actions": list(self.actions),
                "provenance": provenance or {}}
