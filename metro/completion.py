"""Completion certificates for unfinished new lines.

A search cutoff means UNKNOWN and is conservatively masked.
Searches never build assets, charge costs, or evaluate OD.
"""
from metro.objectives import ATOL

COMPLETION_RULE = {
    "name": "certified-new-line",
    "search_state_limit": 4000,
}


class CompletionGuard:
    def __init__(self, core):
        self.core = core
        self.max_states = COMPLETION_RULE["search_state_limit"]
        self.proofs = {}
        self.stats = dict(
            queries=0, certified=0, cache_hits=0, expanded=0,
            rejected_no_completion=0, rejected_search_limit=0,
        )

    def clear(self):
        # Certificates belong to one episode; counters remain cumulative.
        self.proofs.clear()

    def allows(self, resolved):
        core = self.core
        u, v, name, is_new, delta = resolved
        if name in core.scenario.initial_lines:
            return True

        old = next(
            (line for line in core.metro_lines if line.name == name),
            None,
        )
        before = () if is_new else tuple(old.stations)
        if is_new:
            route = (u, v)
            axis = core.node_pos[u] - core.node_pos[v]
        else:
            route = (
                (v,) + before if before[0] == u else before + (v,)
            )
            axis = old.vline

        if len(set(route)) != len(route):
            return False
        if len(route) >= core.min_num:
            return True

        self.stats["queries"] += 1
        others = set(core.edge_build_idx)
        for a, b in zip(before, before[1:]):
            others.discard(core.G[a][b]["idx"])

        groups = {
            node: len(names - {name})
            for node, names in core._memberships.items()
        }
        context = (
            tuple(
                (line.name, tuple(line.stations))
                for line in core.metro_lines if line.name != name
            ),
            name,
            tuple(float(x) for x in axis),
            float(core.budget),
            core.min_num,
            tuple(core.scenario.cost),
        )
        expanded, cutoff = 0, False
        failed = set()

        def moves(path, spent):
            used = others | {
                core.G[a][b]["idx"]
                for a, b in zip(path, path[1:])
            }
            result = {}
            endpoints = (
                (path[0], path[1], 1, True),
                (path[-1], path[-2], -1, False),
            )
            for end, neighbor, sign, at_start in endpoints:
                direction = core.node_pos[end] - core.node_pos[neighbor]
                for node in core.G.neighbors(end):
                    edge = core.G[end][node]
                    vector = core.node_pos[node] - core.node_pos[end]
                    if (
                        node in path or edge["idx"] in used
                        or core._cos(direction, vector) <= .3
                        or sign * core._cos(axis, vector) <= .3
                    ):
                        continue
                    count = groups.get(node, 0)
                    charge = edge["length"] * core.cost_per_kmeter
                    charge += (count == 0) * core.cost_per_station
                    charge += (count == 1) * core.cost_per_trans_station
                    if spent + charge <= core.budget + ATOL:
                        next_path = (
                            (node,) + path if at_start else path + (node,)
                        )
                        # Same node-action resolution as MetroCore:
                        # the last matching endpoint wins.
                        result[node] = (next_path, spent + charge)
            return result

        def valid(path, spent, proof):
            for node in proof:
                options = moves(path, spent)
                if node not in options:
                    return False
                path, spent = options[node]
            return len(path) >= core.min_num

        def search(path, spent):
            nonlocal expanded, cutoff
            if len(path) >= core.min_num:
                return ()

            key = (context, path)
            proof = self.proofs.get(key)
            if proof is not None and valid(path, spent, proof):
                self.stats["cache_hits"] += 1
                return proof
            if path in failed:
                return None
            if expanded >= self.max_states:
                cutoff = True
                return None

            expanded += 1
            options = moves(path, spent)
            ordered = sorted(
                options.items(),
                key=lambda item: (
                    item[1][1], core.node_list_idx[item[0]]
                ),
            )
            for node, (next_path, cost) in ordered:
                proof = search(next_path, cost)
                if proof is not None:
                    answer = (node,) + proof
                    # Cache every suffix to retain certified continuations.
                    self.proofs[key] = answer
                    return answer
            failed.add(path)
            return None

        proof = search(route, core.total_cost + delta)
        self.stats["expanded"] += expanded
        if proof is not None:
            self.stats["certified"] += 1
            return True
        key = (
            "rejected_search_limit" if cutoff
            else "rejected_no_completion"
        )
        self.stats[key] += 1
        return False
