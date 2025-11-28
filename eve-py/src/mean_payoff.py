import itertools
from collections import defaultdict

from igraph import Graph

import parsrml as game_spec
from mp_solver.value_iteration import simple_solve_mp_game


def solve_e_nash_mp(lts: Graph):
    punishments = find_punishment_values(lts)
    print(punishments)

    z_vectors = generate_z_vectors(punishments)
    print(z_vectors)

    for z_vector in z_vectors:
        G_z = compute_G_z(lts, punishments, z_vector)
        print(G_z)

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
        player_puns = punishments[player_name]
        if isinstance(player_puns, dict):
            values = list(player_puns.values())
        else:
            values = [player_puns]
        all_punishment_values.append(sorted(set(values)))

    z_vectors = []
    for combo in itertools.product(*all_punishment_values):
        z_vector = {}
        for i, player_name in enumerate(player_names):
            z_vector[player_name] = combo[i]
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
    owned_vars = get_player_vars_dict()

    for v in g.vs:
        state_label = v["label"][~0] if isinstance(v["label"], list) else v["label"]
        keep_state = True

        for player_name, z_value in z_vector.items():
            try:
                pun_value = punishments[player_name].get(state_label[~0][~0], float('-inf'))
            except (IndexError, TypeError):
                pun_value = punishments[player_name].get(state_label, float('-inf'))

            if pun_value > z_value:
                keep_state = False
                break

        if keep_state:
            valid_states.add(v.index)

    valid_edges = set()
    for v_index in valid_states:
        v = g.vs[v_index]

        for edge_index in g.incident(v, mode="out"):
            edge = g.es[edge_index]
            if not edge["direction"]:
                continue

            is_secure = True

            for player_name, z_value in z_vector.items():
                if not is_action_secure(g, edge, player_name, z_value, punishments, valid_states, owned_vars):
                    is_secure = False
                    break

            if is_secure:
                if edge.target in valid_states:
                    valid_edges.add(edge_index)

    G_z = Graph(directed=True)
    old_to_new_map = {}

    sorted_valid_states = sorted(list(valid_states))
    for new_index, old_index in enumerate(sorted_valid_states):
        old_vertex = g.vs[old_index]
        G_z.add_vertex(**old_vertex.attributes())
        old_to_new_map[old_index] = new_index

    edges_to_add = []
    edge_attributes = defaultdict(list)

    for old_edge_index in valid_edges:
        old_edge = g.es[old_edge_index]

        if old_edge.source in old_to_new_map and old_edge.target in old_to_new_map:
            new_source = old_to_new_map[old_edge.source]
            new_target = old_to_new_map[old_edge.target]

            edges_to_add.append((new_source, new_target))

            for attr_name, attr_val in old_edge.attributes().items():
                edge_attributes[attr_name].append(attr_val)

    if edges_to_add:
        G_z.add_edges(edges_to_add)

        for attr_name, values in edge_attributes.items():
            G_z.es[attr_name] = values

    return G_z


def is_action_secure(g: Graph, edge, player_name: str, z_value: float,
                     punishments: dict, valid_states: set, player_vars_dict: dict) -> bool:
    print(player_vars_dict)
    player_vars = player_vars_dict.get(player_name, set())
    action_profile = edge["direction"]

    for var in player_vars:
        if var in action_profile:
            for deviation in [True, False]:
                deviated_action = list(action_profile)
                deviated_action[deviated_action.index(var)] = not var if deviation else var

                target_state_index = find_transition_target(g, edge.source, deviated_action)
                if target_state_index is None:
                    continue

                if target_state_index not in valid_states:
                    return False

                src = g.vs.find(edge.source)
                state_label = src["label"][~0][~0]
                pun_value = punishments[player_name].get(state_label, float('-inf'))
                if pun_value > z_value:
                    return False

    return True


def find_transition_target(lts: Graph, source_index: int, action_profile: list):
    source_vertex = lts.vs[source_index]

    for edge_index in lts.incident(source_vertex, mode="out"):
        edge = lts.es[edge_index]
        if edge["direction"] == action_profile:
            return edge.target

    return None


def get_player_vars_dict():
    player_vars_dict = {}
    for module in game_spec.modules:
        player_name = list(module[1])[0]
        owned_vars = module[2]
        player_vars_dict[player_name] = owned_vars
    return player_vars_dict
