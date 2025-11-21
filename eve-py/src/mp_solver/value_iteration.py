from typing import List, Dict, Set

from igraph import Graph, Vertex


def simple_solve_mp_game(g):
    values = {v['label'][~0]: 0 for v in g.vs}

    for iteration in range(100):
        old_values = values.copy()

        for v in g.vs:
            v_label = v['label'][~0]
            outgoing_edges = g.es[g.incident(v, mode="out")]

            if v['player'] == 0:
                min_val = float('inf')
                for edge in outgoing_edges:
                    target = g.vs[edge.target]
                    target_label = target['label'][~0]
                    val = edge['weight'] + old_values[target_label]
                    min_val = min(min_val, val)
                values[v_label] = min_val
            else:
                max_val = float('-inf')
                for edge in outgoing_edges:
                    target = g.vs[edge.target]
                    target_label = target['label'][~0]
                    val = edge['weight'] + old_values[target_label]
                    max_val = max(max_val, val)
                values[v_label] = max_val

        if old_values == values:
            print(f"Converged after {iteration} iterations")
            break

    return values


def solve_mp_game_faster(game_graph: Graph):
    if len(game_graph.vs) == 0:
        return {}

    weights = [e['weight'] for e in game_graph.es]
    W = max(abs(w) for w in weights) if weights else 1
    n = len(game_graph.vs)

    values = {}
    solve_mp_game_recursive(game_graph, -W, W, values, n)
    return values


def solve_mp_game_recursive(g, low, high, values, max_denominator):
    if len(g.vs) == 0:
        return

    if len(g.vs) == 1 or high - low < 1.0 / (max_denominator ** 2):
        mid = (low + high) / 2
        for v in g.vs:
            label = v['label'] if not isinstance(v['label'], list) else v['label'][~0]

            if label not in values:
                values[label] = mid

        return

    q1, l1 = find_q1_l1(low, high, max_denominator)
    q2, l2 = find_q2_l2(low, high, max_denominator)

    if q1 is not None and q2 is not None and q1 / l1 == q2 / l2:

        if q1 / l1 == low:
            a1 = low
            a2 = (low + high) / 2
        elif q1 / l1 == high:
            a1 = (low + high) / 2
            a2 = high
        else:
            a1 = q1 / l1
            a2 = min(high, a1 + (high - low) / 4)
    elif q1 is None or q2 is None:
        mid = (low + high) / 2
        for v in g.vs:
            label = v['label'] if not isinstance(v['label'], list) else v['label'][~0]
            if label not in values:
                values[label] = mid
        return
    else:
        a1 = q1 / l1
        a2 = q2 / l2

    def get_winning_vertices(energy_result: Dict) -> Set[Vertex]:
        winning_vertices = set()
        maximal_mg = find_maximal_progress_measure(g)
        for label, progress_value in energy_result.items():
            for v in g.vs:
                if v['label'] == label and progress_value != maximal_mg:
                    winning_vertices.add(v)
                    break
        return winning_vertices

    V_ge_a1 = get_winning_vertices(solve_energy_game(reweight_subgame(g, l1, q1)))
    V_le_a1 = get_winning_vertices(solve_energy_game(inverse_players(reweight_subgame(g, -1 * l1, q1))))
    V_ge_a2 = get_winning_vertices(solve_energy_game(reweight_subgame(g, l2, q2)))
    V_le_a2 = get_winning_vertices(solve_energy_game(inverse_players(reweight_subgame(g, -1 * l2, q2))))

    V_eq_a1 = V_ge_a1.intersection(V_le_a1)
    V_eq_a2 = V_ge_a2.intersection(V_le_a2)
    V_lt_a1 = set(g.vs).difference(V_ge_a1)
    V_gt_a2 = set(g.vs).difference(V_le_a2)

    for v in V_eq_a1:
        values[v['label']] = a1
    for v in V_eq_a2:
        values[v['label']] = a2

    if not V_lt_a1 and not V_gt_a2 and len(V_eq_a1) == len(g.vs):

        for v in g.vs:
            label = v['label']
            if label not in values:
                values[label] = a1
        return

    if V_lt_a1:
        subgraph_lt = build_subgraph(g, V_lt_a1)
        solve_mp_game_recursive(subgraph_lt, low, a1, values, min(max_denominator, len(subgraph_lt.vs)))

    if V_gt_a2:
        subgraph_gt = build_subgraph(g, V_gt_a2)
        solve_mp_game_recursive(subgraph_gt, a2, high, values, min(max_denominator, len(subgraph_gt.vs)))

    for v in g.vs:
        label = v['label'] if not isinstance(v['label'], list) else v['label'][~0]
        if label not in values:
            mid = (a1 + a2) / 2
            values[label] = mid


def reweight_subgame(game_graph: Graph, l, q):
    g = game_graph.copy()
    for e in g.es:
        e['weight'] = l * e['weight'] - q
    return g


def inverse_players(game_graph: Graph):
    g = game_graph.copy()
    for v in g.vs:
        v['player'] = 1 - v['player']
    return g


def find_q1_l1(r_i, s_i, max_denominator):
    target = (r_i + s_i) / 2
    best_q = None
    best_l = None
    best_value = -float('inf')

    for l in range(1, max_denominator + 1):
        q = int(target * l)

        while q / l > target:
            q -= 1

        if q / l < r_i:
            continue

        value = q / l

        if value > best_value:
            best_value = value
            best_q = q
            best_l = l

    return best_q, best_l


def find_q2_l2(r_i, s_i, max_denominator):
    target = (r_i + s_i) / 2
    best_q = None
    best_l = None
    best_value = float('inf')

    for l in range(1, max_denominator + 1):
        q = int(target * l)

        while q / l < target:
            q += 1

        if q / l > s_i:
            continue

        value = q / l

        if value < best_value:
            best_value = value
            best_q = q
            best_l = l

    return best_q, best_l


def build_subgraph(original_graph: Graph, vertices_to_keep: Set[Vertex]) -> Graph:
    if not vertices_to_keep:
        return Graph()

    vertex_indices = [v.index for v in vertices_to_keep]

    subgraph = original_graph.subgraph(vertex_indices)

    for i, v in enumerate(subgraph.vs):
        original_v = original_graph.vs[vertex_indices[i]]
        for attr in original_v.attributes():
            v[attr] = original_v[attr]

    for i, e in enumerate(subgraph.es):
        original_e = original_graph.es[subgraph.get_eid(e.source, e.target)]
        for attr in original_e.attributes():
            e[attr] = original_e[attr]

    return subgraph


def solve_energy_game(game_graph: Graph):
    progress_measures: Dict[Vertex, int] = {v: 0 for v in game_graph.vs}
    maximal_mg = find_maximal_progress_measure(game_graph)

    l: List[Vertex] = []
    v_count: Dict[Vertex, int] = {}

    for v in game_graph.vs:
        if v["player"] == 0:
            consistent_edges = 0
            for e in v.out_edges():
                target_vertex = game_graph.vs[e.target]
                required = plimsol(progress_measures[target_vertex], e["weight"], maximal_mg)
                if progress_measures[v] >= required:
                    consistent_edges += 1
            v_count[v] = consistent_edges
            if consistent_edges == 0:
                l.append(v)
        else:
            all_consistent = True
            for e in v.out_edges():
                target_vertex = game_graph.vs[e.target]
                required = plimsol(progress_measures[target_vertex], e["weight"], maximal_mg)
                if progress_measures[v] < required:
                    all_consistent = False
                    break
            v_count[v] = 1 if all_consistent else 0
            if not all_consistent:
                l.append(v)

    iteration = 0
    max_iterations = len(game_graph.vs) * maximal_mg * 2
    changed = True
    while l and iteration < max_iterations and changed:
        iteration += 1
        changed = False
        current_l = l.copy()
        l.clear()

        for v in current_l:
            old_value = progress_measures[v]

            progress_measures = delta(v, game_graph, progress_measures, maximal_mg)
            new_value = progress_measures[v]

            if new_value != old_value:
                changed = True

                for pred in game_graph.vs:
                    edge_id = game_graph.get_eid(pred.index, v.index, directed=True, error=False)
                    if edge_id != -1:
                        edge_weight = game_graph.es[edge_id]["weight"]
                        new_required = plimsol(new_value, edge_weight, maximal_mg)

                        if pred["player"] == 1:

                            if progress_measures[pred] < new_required:
                                if pred not in l:
                                    l.append(pred)

                        else:

                            consistent_count = 0
                            for e in pred.out_edges():
                                target = game_graph.vs[e.target]
                                req = plimsol(progress_measures[target], e["weight"], maximal_mg)
                                if progress_measures[pred] >= req:
                                    consistent_count += 1

                            v_count[pred] = consistent_count
                            if consistent_count == 0 and pred not in l:
                                l.append(pred)

    winning_vertices = set()

    for v in game_graph.vs:

        if progress_measures[v] < maximal_mg:
            winning_vertices.add(v)

    result_by_label = {}
    for vertex, value in progress_measures.items():
        if isinstance(vertex['label'], list):
            vertex['label'] = vertex['label'][~0]

        result_by_label[vertex['label']] = value

    return result_by_label


def find_maximal_progress_measure(g: Graph) -> int:
    max_abs_weight = 0
    for e in g.es:
        if abs(e["weight"]) > max_abs_weight:
            max_abs_weight = abs(e["weight"])
    maximal_mg = len(g.vs) * max_abs_weight
    return maximal_mg


def plimsol(a, b, maximal_mg: int):
    if a == maximal_mg:
        return maximal_mg

    result = a - b
    if result < 0:
        return 0
    elif result > maximal_mg:
        return maximal_mg
    else:
        return result


def delta(v, g: Graph, f: Dict[Vertex, int], maximal_mg: int) -> Dict[Vertex, int]:
    lifted_f = f.copy()

    if v["player"] == 0:
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
