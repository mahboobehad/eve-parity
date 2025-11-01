import itertools
import os
import subprocess
import tempfile
from collections import defaultdict

from igraph import Graph

import parsrml as game_spec


def create_zero_sum_games(lts: Graph):
    players = [(list(m[1])[0], m[2]) for m in game_spec.modules]
    zero_sum_turn_based_games = []
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
                print(f"{e_source['label']} -- (l={edge['label']}, {edge['weight']}) --> {e_target['label']}")

        solve_game_with_meanpayoff(g, name)
        break


def create_player_game(lts: Graph, player_name: str, player_vars: set):
    g = Graph(directed=True)

    intermediate_vs_from_v = defaultdict(set)
    for v in lts.vs:
        v_payoff = get_state_payoff(player_name, v)

        g.add_vertex(label=v["label"][~0], type="state", payoff=v_payoff, player="min")
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
                        player="max"
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

def solve_game_with_meanpayoff(g: Graph, player_name: str):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_filename = f.name
        edges_written = convert_game_to_meanpayoff_format(g, f)

    print(f"Created game file with {edges_written} edges: {temp_filename}")

    print("Game file contents:")
    with open(temp_filename, 'r') as f:
        print(f.read())

    try:
        current_dir = os.getcwd()
        meanpayoff_path = os.path.join(current_dir, 'meanpayoff')

        print(f"Running: {meanpayoff_path} {temp_filename}")

        result = subprocess.run([meanpayoff_path, temp_filename],
                                capture_output=True, text=True, timeout=30,
                                cwd=current_dir)

        print("Meanpayoff solver completed")
        print(f"Return code: {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # Parse the results
        if result.returncode == 0:
            parse_meanpayoff_results(result.stdout, player_name)
        else:
            print(f"Solver failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        print(f"Meanpayoff solver timed out for player {player_name}")
    except FileNotFoundError:
        print(f"Error: meanpayoff executable not found at {meanpayoff_path}")
        print("Current directory contents:")
        try:
            print(os.listdir(current_dir))
        except:
            pass
    except Exception as e:
        print(f"Error running meanpayoff solver: {e}")
    finally:
        try:
            os.unlink(temp_filename)
        except:
            pass


def convert_game_to_meanpayoff_format(g: Graph, file_handle):
    edges_written = 0

    vertex_mapping = {}
    min_index = 0
    max_index = 1
    for i, vertex in enumerate(g.vs):
        if vertex['player'] == "min":
            index = min_index
            min_index += 2
        else:
            index = max_index
            max_index += 2
        vertex_mapping[vertex['label'][~0]] = index

    written_edges = set()
    for edge in g.es:
        source_id = vertex_mapping[g.vs[edge.source]['label'][~0]]
        target_id = vertex_mapping[g.vs[edge.target]['label'][~0]]

        if source_id == target_id:
            print(f"  Skipping self-loop: {source_id} -> {target_id}")
            continue

        edge_key = (source_id, target_id)
        if edge_key in written_edges:
            print(f"  Skipping duplicate edge: {source_id} -> {target_id}")
            continue

        weight = edge["weight"]

        file_handle.write(f"{source_id} {target_id} {weight}\n")
        written_edges.add(edge_key)
        edges_written += 1
        print(f"  Writing edge: {source_id} -> {target_id} (weight: {weight})")

    return edges_written


def parse_meanpayoff_results(output: str, player_name: str):
    print(f"Raw output for {player_name}:")
    print("---")
    print(output)
    print("---")

    lines = output.split('\n')

    results_section = False
    state_results = []

    for line in lines:
        print(f"Processing line: '{line}'")
        if "Results:" in line:
            results_section = True
            print("Found results section")
            continue
        if results_section and "State:" in line:
            print(f"Found state result line: {line}")
            # Parse lines like: "State: 0  MP: 3.0  Bias: 1.5"
            parts = line.split()
            if len(parts) >= 6:
                try:
                    state_id = int(parts[1])
                    mp_value = float(parts[3])
                    bias_value = float(parts[5])
                    state_results.append((state_id, mp_value, bias_value))
                    print(f"Successfully parsed: state={state_id}, mp={mp_value}, bias={bias_value}")
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {e}")

    print(f"\nParsed results for {player_name}:")
    if state_results:
        for state_id, mp_value, bias_value in state_results:
            print(f"  State {state_id}: MP={mp_value}, Bias={bias_value}")
    else:
        print("  No results parsed - check if solver produced expected output format")

    return state_results
