from functools import cache
from typing import List, Dict

from igraph import Graph, Vertex


def solve_energy_game(game_graph: Graph):
    l: List[Vertex] = list()
    for v in game_graph.vs.select(player="0"):
        is_all_negative = True
        for e in v.out_edges():
            if e["weight"] > 0:
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
    for v in game_graph.vs:
        progress_measures[v] = 0
        if v["player"] == "0":
            if v in l:
                v_count[v] = 0
            else:
                # TODO: implement post loop
                pass

    while len(l) > 0:
        v = l.pop()
        pass


def plimsol(a, b, g: Graph):
    mg = 0

    maximal_mg = find_maximal_progress_measure(g)
    for e in g.es:
        mg += max(0, -1 * e["weight"])
    if a != maximal_mg and a - b < mg:
        return max(0, a - b)
    else:
        return maximal_mg


@cache
def find_maximal_progress_measure(g):
    max_weight = 0
    for e in g.es:
        if e["weight"] > max_weight:
            max_weight = e["weight"]
    maximal_mg = len(g.vs) * max_weight
    return maximal_mg
