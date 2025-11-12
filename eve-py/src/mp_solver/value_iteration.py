from typing import List, Dict

from igraph import Graph, Vertex


def solve_energy_game(game_graph: Graph):
    l: List[Vertex] = list()
    for v in game_graph.vs.select(player="0"):
        is_all_negative = True
        for e in v.out_edges():
            if e["weight"] >= 0:
                is_all_negative = False
                break
        if is_all_negative:
            l.append(v)

    for v in game_graph.vs.select(player="1"):
        is_any_negative = False
        for e in v.out_edges():
            if e["weight"] < 0:
                is_any_negative = True
                break
        if is_any_negative:
            l.append(v)

    progress_measures: Dict[Vertex, int] = {v: 0 for v in game_graph.vs}
    v_count: Dict[Vertex, int] = dict()

    maximal_mg = find_maximal_progress_measure(game_graph)

    for v in game_graph.vs:
        progress_measures[v] = 0
        if v in l:
            v_count[v] = 0
        else:
            counter = 0
            for e in v.out_edges():
                target_vertex = game_graph.vs[e.target]
                if progress_measures[v] >= plimsol(progress_measures[target_vertex], e["weight"], maximal_mg):
                    counter += 1
            v_count[v] = counter

    while len(l) > 0:
        v = l.pop()
        old_value = progress_measures[v]
        progress_measures = delta(v, game_graph, progress_measures, maximal_mg)

        if v["player"] == "0":
            counter = 0
            for e in v.out_edges():
                target_vertex = game_graph.vs[e.target]
                if progress_measures[v] >= plimsol(progress_measures[target_vertex], e["weight"], maximal_mg):
                    counter += 1
            v_count[v] = counter

        new_value = progress_measures[v]

        if new_value != old_value:
            for pred_vertex in game_graph.vs:
                edge_id = game_graph.get_eid(pred_vertex.index, v.index, directed=True, error=False)
                if edge_id != -1:
                    edge_weight = game_graph.es[edge_id]["weight"]

                    if pred_vertex["player"] == "1":
                        if progress_measures[pred_vertex] < plimsol(new_value, edge_weight, maximal_mg):
                            if pred_vertex not in l:
                                l.append(pred_vertex)
                    else:
                        if progress_measures[pred_vertex] < plimsol(new_value, edge_weight, maximal_mg):
                            v_count[pred_vertex] -= 1
                            if v_count[pred_vertex] == 0 and pred_vertex not in l:
                                l.append(pred_vertex)

    result_by_label = {}
    for vertex, value in progress_measures.items():
        result_by_label[vertex['label']] = value

    return result_by_label


def plimsol(a, b, maximal_mg: int):
    if a == maximal_mg:
        return maximal_mg

    result = a - b
    if result <= maximal_mg:
        return max(0, result)
    else:
        return maximal_mg


def find_maximal_progress_measure(g: Graph) -> int:
    max_weight = 0
    for e in g.es:
        if e["weight"] > max_weight:
            max_weight = e["weight"]
    maximal_mg = len(g.vs) * max_weight
    return maximal_mg


def delta(v, g: Graph, f: Dict[Vertex, int], maximal_mg: int) -> Dict[Vertex, int]:
    lifted_f = f.copy()

    if v["player"] == "0":
        min_value = None

        for e in v.out_edges():
            target_vertex = g.vs[e.target]
            candidate = plimsol(f[target_vertex], e["weight"], maximal_mg)

            if min_value is None or candidate < min_value:
                min_value = candidate

        lifted_f[v] = min_value if min_value is not None else maximal_mg

    else:
        max_value = None

        for e in v.out_edges():
            target_vertex = g.vs[e.target]
            candidate = plimsol(f[target_vertex], e["weight"], maximal_mg)

            if max_value is None or candidate > max_value:
                max_value = candidate

        lifted_f[v] = max_value if max_value is not None else maximal_mg

    return lifted_f
