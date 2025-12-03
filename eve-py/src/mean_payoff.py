import itertools
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

from igraph import Graph

import parsrml as game_spec
from mp_solver.value_iteration import simple_solve_mp_game
from srmlutil import productInit


def solve_e_nash_mp(lts: Graph):
    punishments = find_punishment_values(lts)
    print(punishments)

    z_vectors = generate_z_vectors(punishments)
    print(z_vectors)

    for z_vector in z_vectors:
        G_z = compute_G_z(lts, punishments, z_vector)
        print(G_z)
        # TODO: call lim avg ltl checker

    return False


def solve_a_nash_mp(lts: Graph):
    punishments = find_punishment_values(lts)

    z_vectors = generate_z_vectors(punishments)
    print(z_vectors)

    for z_vector in z_vectors:
        G_z = compute_G_z(lts, punishments, z_vector)
        print(G_z)
        # TODO: call lim avg ltl checker
    return False


def find_punishment_values(lts: Graph):
    players = [(list(m[1])[0], m[2]) for m in game_spec.modules]
    zero_sum_turn_based_games = []

    punishments = dict()
    for name, owned_vars in players:
        g = create_player_game(lts, name, owned_vars)
        zero_sum_turn_based_games.append(g)

        print(f"{name}'s game is:")
        print(f"vs: {len(g.vs)}")
        print(f"es: {len(g.es)}")
        for v in g.vs:
            for e_index in g.incident(v, mode="out"):
                edge = g.es[e_index]
                e_source = g.vs[edge.source]
                e_target = g.vs[edge.target]
                print(
                    f"{e_source['label'], e_source['player']} -- (l={edge['label']}, {edge['weight']}) --> {e_target['label'], e_target['player']}"
                )

        values = solve_mean_payoff_game(g)
        punishments[name] = values

    return punishments


def generate_z_vectors(punishments):
    player_names = list(punishments.keys())
    all_punishment_values = []

    for player_name in player_names:
        vals = set(punishments[player_name].values())
        all_punishment_values.append(sorted(list(vals)))

    z_vectors = []
    for combo in itertools.product(*all_punishment_values):
        z_vector = {name: val for name, val in zip(player_names, combo)}
        z_vectors.append(z_vector)

    return z_vectors


def create_player_game(lts: Graph, player_name: str, player_vars: set):
    g = Graph(directed=True)

    intermediate_vs_from_v = defaultdict(set)
    for v in lts.vs:
        v_payoff = get_state_payoff(player_name, v)

        g.add_vertex(label=v["label"][~0], type="state", payoff=v_payoff, player=1)
        source = g.vs.find(label=v["label"][~0])

        coalition_actions_from_v = set()
        for edge_index in lts.incident(v, mode="out"):
            out_edge = lts.es[edge_index]
            if not out_edge["direction"]:
                continue

            for var in out_edge["direction"]:
                if var and var not in player_vars:
                    coalition_actions_from_v.add(var)

        if coalition_actions_from_v:
            actions_list = list(sorted(coalition_actions_from_v))

            i = 0
            for combo in itertools.product([True, False], repeat=len(actions_list)):
                full_node_name = f"s{v.index}_{actions_list[i]}_{str(combo[i]).lower()}"
                try:
                    existing_node = g.vs.find(label=full_node_name)
                    intermediate_vertex = existing_node
                except ValueError:
                    g.add_vertex(
                        label=[full_node_name],
                        type="intermediate",
                        payoff=v_payoff,
                        player=0
                    )
                    intermediate_vs_from_v[v].add(full_node_name)
                    intermediate_vertex = g.vs.find(label=[full_node_name])

                add_edge = True
                if g.es:
                    matched_edge = g.es.find(label="min")
                    if matched_edge.source == source.index and matched_edge.target == intermediate_vertex.index:
                        add_edge = False
                if add_edge:
                    g.add_edge(
                        source=source,
                        target=intermediate_vertex,
                        label="min",
                        weight=v_payoff * -1
                    )

    for v in lts.vs:
        source = g.vs.find(label=v["label"][~0])
        v_payoff = get_state_payoff(player_name, v)

        for edge_index in lts.incident(v, mode="out"):
            out_edge = lts.es[edge_index]
            if not out_edge["direction"]:
                continue
            for var in out_edge["direction"]:
                if var and var in player_vars:
                    for i_name in intermediate_vs_from_v[v]:
                        intermediate_v = g.vs.find(label=[i_name])
                        if out_edge.source == out_edge.target:
                            g.add_edge(
                                source=intermediate_v,
                                target=source,
                                label="max",
                                weight=v_payoff
                            )
                        else:
                            target = g.vs.find(label=lts.vs[out_edge.target]["label"][~0])
                            g.add_edge(
                                source=intermediate_v,
                                target=target,
                                label="max",
                                weight=v_payoff
                            )
    return g


def get_state_payoff(player_name, v):
    v_payoff = None
    for payoff_spec in game_spec.payoffs:
        player = payoff_spec["module"]
        value = payoff_spec["value"]
        state = payoff_spec["state"]
        if player == player_name and state in v["label"][~0]:
            v_payoff = value
            break
    return v_payoff


def solve_mean_payoff_game(g: Graph):
    game_values = simple_solve_mp_game(g)
    print(game_values)

    for v in g.vs:
        if v["player"] == 0:
            game_values.pop(v["label"][~0])
    return game_values


def compute_G_z(g, punishments, z_vector):
    valid_states = set()

    for v in g.vs:
        is_valid = True
        label = v["label"]

        for player_name, z_value in z_vector.items():
            pun_value = get_punishment_value(label, punishments[player_name])

            if pun_value > z_value:
                is_valid = False
                break

        if is_valid:
            valid_states.add(v.index)

    G_z = Graph(directed=True)
    old_to_new_map = {}

    sorted_valid_states = sorted(list(valid_states))
    for new_index, old_index in enumerate(sorted_valid_states):
        old_vertex = g.vs[old_index]
        G_z.add_vertex(**old_vertex.attributes())
        old_to_new_map[old_index] = new_index

    edges_to_add = []
    edge_attributes = defaultdict(list)

    for edge in g.es:
        if edge.source in valid_states and edge.target in valid_states:
            new_source = old_to_new_map[edge.source]
            new_target = old_to_new_map[edge.target]
            edges_to_add.append((new_source, new_target))

            for attr_name, attr_val in edge.attributes().items():
                edge_attributes[attr_name].append(attr_val)

    if edges_to_add:
        G_z.add_edges(edges_to_add)
        for attr_name, values in edge_attributes.items():
            G_z.es[attr_name] = values

    return G_z


def get_punishment_value(label_data, punishment_dict):
    candidates = []

    if isinstance(label_data, (list, tuple)):
        stack = list(label_data)
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
            else:
                candidates.append(item)
    else:
        candidates.append(label_data)

    for cand in candidates:
        if str(cand) in punishment_dict:
            return punishment_dict[str(cand)]
        if cand in punishment_dict:
            return punishment_dict[cand]

    return float('inf')


def run_limavg_checker(qks_dict, formula, quiet=True):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(qks_dict, f, indent=2)
        qks_path = f.name

    tool_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ltl_lim_avg_checker",
        "core",
        "main.py"
    )
    tool_path = os.path.abspath(tool_path)

    cmd = [sys.executable, tool_path, "--qks-file", qks_path]
    if quiet:
        cmd.append("--quiet")
    cmd.append(formula)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()

    try:
        os.remove(qks_path)
    except:
        pass

    return proc.returncode, out, err


def qks_from_igraph(G, mdl, payoff_key="payoff"):
    init_prod = list(productInit(mdl))
    if len(init_prod) != 1:
        raise ValueError(f"SRML model has {len(init_prod)} initial states; QKS supports exactly one.")

    init_true_vars = set(init_prod[0])

    states = []
    edges = []
    logical_formulas = {}
    numeric_values = {}
    boolean_vars = set()

    def flatten_label(label):
        if isinstance(label, (list, tuple)):
            flat = []
            stack = list(label)
            while stack:
                x = stack.pop()
                if isinstance(x, (list, tuple)):
                    stack.extend(x)
                else:
                    flat.append(str(x))
            return flat
        return [str(label)]

    init_state_name = None

    for v in G.vs:
        lbl = flatten_label(v["label"])
        state_name = f"q{v.index}"
        states.append(state_name)

        valuation = set([x for x in lbl if not x.endswith("_false")])
        logical_formulas[state_name] = list(valuation)
        boolean_vars |= valuation

        if payoff_key in v.attributes():
            numeric_values[state_name] = {"payoff": float(v[payoff_key])}
        else:
            numeric_values[state_name] = {"payoff": 0.0}

        if valuation == init_true_vars:
            init_state_name = state_name

    if init_state_name is None:
        raise ValueError("ERROR: No igraph vertex matches the SRML initial valuation.")

    for e in G.es:
        s = f"q{e.source}"
        t = f"q{e.target}"
        edges.append((s, t))

    return {
        "states": states,
        "init_state": init_state_name,
        "edges": edges,
        "boolean_vars": sorted(boolean_vars),
        "logical_formulas": logical_formulas,
        "numeric_values": numeric_values
    }
