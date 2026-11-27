"""Independent final-network evaluation, independent of environment caches."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import pickle
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "metro-network"
RTOL, ATOL = 1e-7, 1e-8


def validate_weights(weights):
    values = np.asarray(weights, dtype=np.float64)
    if (values.shape != (3,) or not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isclose(values.sum(), 1., rtol=0, atol=1e-8)):
        raise ValueError(
            "Three finite nonnegative normalized coefficients required")
    return tuple(float(v) for v in values)


def weight_label(weights):
    return "w" + "_".join(format(v, ".12g") for v in validate_weights(weights))


def json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=json_default,
                                     ensure_ascii=False).encode()).hexdigest()


def geographic_distances(positions):
    xy = np.radians(np.asarray(positions, dtype=np.float64))
    lat, lon = xy[:, 0], xy[:, 1]
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :])
         * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def memberships(lines):
    result = {}
    for name, route in lines.items():
        for node in set(route):
            result.setdefault(node, set()).add(name)
    return result


def route_edges(lines):
    return {frozenset((u, v)) for route in lines.values()
            for u, v in zip(route[:-1], route[1:]) if u != v}


def implementation_hash():
    paths = sorted(p.relative_to(ROOT) for directory in ("metro", "khrylib", "ours")
                   for p in (ROOT / directory).rglob("*.py"))
    paths += [Path("metro/cfg/morl.yaml"), Path("requirements.txt")]
    return digest({str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})


@dataclass(frozen=True)
class ObjectiveSpec:
    od_scale: float
    cost_scale: float = 60.0
    names: tuple = ("od_gain", "access_gain", "construction_cost")
    directions: tuple = (1, 1, -1)

    def __post_init__(self):
        if not np.isfinite(self.od_scale) or self.od_scale <= 0:
            raise ValueError("Positive frozen OD calibration scale required")
        if self.cost_scale != 60:
            raise ValueError("Cost scale is fixed")

    def utility(self, gain, access, cost, weights):
        w = validate_weights(weights)
        return float(w[0] * gain / self.od_scale
                     + w[1] * access
                     - w[2] * cost / self.cost_scale)

    @classmethod
    def from_metadata(cls, metadata):
        scales = metadata.get("scales", [])
        if len(scales) != 3 or scales[1] != 1:
            raise ValueError("Three-objective scales required")
        spec = cls(scales[0], scales[2])
        if spec.metadata() != metadata:
            raise ValueError("Objective metadata mismatch")
        return spec

    def metadata(self):
        return {"version": VERSION, "names": list(self.names),
                "directions": list(self.directions), "scales": [self.od_scale, 1., self.cost_scale]}


class Scenario:
    def __init__(self, city, pre_line=4, cost=(.5, .3, .3), data_root=None):
        self.city, self.pre_line = city, pre_line
        self.cost = tuple(float(c) for c in cost)
        if len(self.cost) != 3 or any(c < 0 or not np.isfinite(c) for c in self.cost):
            raise ValueError("Three finite nonnegative cost coefficients required")
        folder = Path(data_root or ROOT / "data") / city / f"{pre_line}lines"
        loaded, hashes = {}, {}
        for name in ("city_graph", "region_processed", "od_pair", "line_info"):
            path = folder / f"{name}.pickle"
            raw = path.read_bytes()
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
            loaded[name] = pickle.loads(raw)
        self.graph = nx.Graph(loaded["city_graph"])
        self.regions, self.od = loaded["region_processed"], loaded["od_pair"]
        names, routes = loaded["line_info"]
        self.initial_lines = {name: list(routes[name]) for name in names}
        self.line_info = [list(names), self.initial_lines]
        self.nodes = list(self.graph)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.node_lookup = {str(n): n for n in self.nodes}
        if len(self.node_lookup) != len(self.nodes):
            raise ValueError("Ambiguous serialized node IDs")
        self.positions = np.array([self.graph.nodes[n]["pos"] for n in self.nodes])
        if not np.isfinite(self.positions).all():
            raise ValueError("Nonfinite coordinates")
        if any(not np.isfinite(d["length"]) or d["length"] < 0
               for _, _, d in self.graph.edges(data=True)):
            raise ValueError("Invalid edge length")
        if any(n not in self.regions or n not in self.od for n in self.nodes):
            raise ValueError("Missing region or OD row")
        self.direct = geographic_distances(self.positions)
        od = np.array([[self.od[u][v] for v in self.nodes] for u in self.nodes], dtype=np.float64)
        if not np.isfinite(od).all() or (od < 0).any():
            raise ValueError("Invalid OD counts")
        np.fill_diagonal(od, 0)
        self.flow = np.triu(od + od.T, k=1) / 1e5
        self.total_demand = float(self.flow.sum())
        self.initial_memberships = memberships(self.initial_lines)
        self.initial_edges = route_edges(self.initial_lines)
        self.initial_distance, self.initial_active = self.distances(self.initial_lines)
        self.initial_score = self.score(self.initial_distance, self.initial_active)
        self.initial_nearest = self.nearest_distances(self.initial_active)
        self.initial_access_distance = float(self.initial_nearest.mean())
        if (not np.isfinite(self.initial_access_distance)
                or self.initial_access_distance <= 1e-8):
            raise ValueError("Initial mean access distance must exceed 1e-8 km")
        self.manifest = {
            "city": city, "pre_line": pre_line, "files": hashes,
            "nodes": [str(n) for n in self.nodes], "node_id_type": type(self.nodes[0]).__name__,
            "initial_lines": self.encode_lines(self.initial_lines), "cost_coefficients": self.cost,
            "objective_version": VERSION, "candidate_nodes": len(self.nodes),
            "candidate_edges": self.graph.number_of_edges(), "regions": len(self.regions),
            "initial_stations": int(self.initial_active.sum()),
            "position_min": self.positions.min(axis=0).tolist(),
            "position_max": self.positions.max(axis=0).tolist(),
            "feature_widths": sorted({len(r["feature"]) for r in self.regions.values()}),
            "zero_length_edges": [[str(u), str(v)] for u, v, d in self.graph.edges(data=True)
                                  if d["length"] == 0],
            "initial_od_score": self.initial_score, "candidate_od_total": self.total_demand,
            "initial_access_distance_km": self.initial_access_distance,
            "access_definition": "equal-candidate-mean-nearest-geodesic"}
        self.hash = digest(self.manifest)

    def nearest_distances(self, active):
        if not np.any(active):
            raise ValueError("Access requires at least one built station")
        return self.direct[:, np.flatnonzero(active)].min(axis=1)

    def access_metrics(self, active):
        distance = float(self.nearest_distances(active).mean())
        return {
            "initial_access_distance_km": self.initial_access_distance,
            "access_distance_km": distance,
            "access_gain": (
                self.initial_access_distance - distance
            ) / self.initial_access_distance,
        }

    def encode_lines(self, lines):
        return {str(k): [str(n) for n in v] for k, v in lines.items()}

    def decode_lines(self, lines):
        return {k: [self.node_lookup[str(n)] for n in v] for k, v in lines.items()}

    def distances(self, lines):
        graph = nx.Graph()
        for route in lines.values():
            graph.add_nodes_from(route)
            for u, v in zip(route[:-1], route[1:]):
                if u != v:
                    graph.add_edge(u, v, length=float(self.graph[u][v]["length"]))
        distance = np.full((len(self.nodes), len(self.nodes)), np.inf)
        np.fill_diagonal(distance, 0)
        active = np.array([n in graph for n in self.nodes], dtype=bool)
        for u, row in nx.all_pairs_dijkstra_path_length(graph, weight="length"):
            for v, length in row.items():
                distance[self.index[u], self.index[v]] = length
        return distance, active

    def score(self, distance, active):
        ids = np.flatnonzero(active)
        block = np.ix_(ids, ids)
        shortest, direct, flow = distance[block], self.direct[block], self.flow[block]
        ratio = np.divide(direct, shortest, out=np.zeros_like(direct), where=shortest > 0)
        weight = np.where(direct <= 3, 1., np.clip(ratio, .2, .8))
        weight[~np.isfinite(shortest)] = 0
        return float(np.sum(flow * weight))

    def asset_cost(self, lines):
        a, b, c = self.cost
        added = route_edges(lines) - self.initial_edges
        length = sum(self.graph[tuple(e)[0]][tuple(e)[1]]["length"] for e in added)
        groups = memberships(lines)
        stations = set(groups) - set(self.initial_memberships)
        transfers = [n for n, ls in groups.items() if len(ls) >= 2
                     and len(self.initial_memberships.get(n, ())) < 2]
        return float(a * length + b * len(stations) + c * len(transfers))

    def evaluate(self, lines, budget=60, max_new_lines=3, min_stations=10):
        errors = []
        for name, route in self.initial_lines.items():
            final = lines.get(name, [])
            if not any(final[i:i+len(route)] in (route, route[::-1])
                       for i in range(max(0, len(final)-len(route)+1))):
                errors.append(f"initial_line_changed:{name}")
        added = {k: v for k, v in lines.items() if k not in self.initial_lines}
        if len(added) > max_new_lines:
            errors.append("too_many_new_lines")
        for name, route in lines.items():
            if len(route) < 2 or any(n not in self.index for n in route):
                errors.append(f"invalid_route:{name}")
                continue
            if name in added and (len(set(route)) < min_stations or len(set(route)) != len(route)):
                errors.append(f"incomplete_or_repeated_new_line:{name}")
            if any(u == v or not self.graph.has_edge(u, v) for u, v in zip(route[:-1], route[1:])):
                errors.append(f"invalid_edge:{name}")
        if any(e.startswith(("invalid_route", "invalid_edge")) for e in errors):
            return {"feasible": False, "errors": errors}
        distance, active = self.distances(lines)
        score, cost = self.score(distance, active), self.asset_cost(lines)
        if cost > budget + ATOL:
            errors.append("over_budget")
        return {"feasible": not errors, "errors": errors, "od_score": score,
                "od_gain": score - self.initial_score, "cost": cost,
                **self.access_metrics(active),
                "remaining_budget": float(budget - cost),
                "station_count": int(active.sum()), "new_line_count": len(added)}


@lru_cache(maxsize=8)
def load_scenario(city, pre_line=4, cost=(.5, .3, .3), data_root=None):
    return Scenario(city, pre_line, cost, data_root)
