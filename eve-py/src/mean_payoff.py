import itertools
import os
import subprocess
import tempfile
from collections import defaultdict

from igraph import Graph, plot

import parsrml as game_spec


def create_zero_sum_games(lts: Graph):
    players = [(list(m[1])[0], m[2]) for m in game_spec.modules]

    # print("lts is:")
    # for v in lts.vs:
    #     print(v)
    #     for e_index in lts.incident(v):
    #         edge = lts.es[e_index]
    #         e_source = lts.vs[edge.source]
    #         e_target = lts.vs[edge.target]
    #         print(f"{e_source['label'][1]} -- l={edge['direction']} --> {e_target['label'][1]}")

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
                print(f"{e_source['label']} -- l={edge['label']} --> {e_target['label']}")

        # solve_game_with_meanpayoff(g, name)
        # plot_game(g, name)

        break


def create_player_game(lts: Graph, player_name: str, player_vars: set):
    g = Graph(directed=True)

    intermediate_vs_from_v = defaultdict(set)
    for v in lts.vs:
        v_payoff = None
        for payoff_spec in game_spec.payoffs:
            player = payoff_spec["module"]
            value = payoff_spec["value"]
            state = payoff_spec["state"]
            if player == player_name and state in v["label"][~0]:
                v_payoff = value
                break

        g.add_vertex(label=v["label"][~0], type="state", payoff=v_payoff)
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
                full_node_name = f"v{v.index}_{actions_list[i]}_{str(combo[i]).lower()}"
                try:
                    existing_node = g.vs.find(label=full_node_name)
                    intermediate_vertex = existing_node
                except ValueError:
                    g.add_vertex(label=full_node_name,
                                 type="intermediate",
                                 payoff=v_payoff)
                    intermediate_vs_from_v[v].add(full_node_name)
                    intermediate_vertex = g.vs.find(label=full_node_name)

                add_edge = True
                if g.es:
                    matched_edge = g.es.find(label="min")
                    if matched_edge.source == source.index and matched_edge.target == intermediate_vertex.index:
                        add_edge = False
                if add_edge:
                    g.add_edge(source=source,
                               target=intermediate_vertex,
                               label="min")

    for v in lts.vs:
        source = g.vs.find(label=v["label"][~0])
        for edge_index in lts.incident(v, mode="out"):
            out_edge = lts.es[edge_index]
            if not out_edge["direction"]:
                continue
            for var in out_edge["direction"]:
                if var and var in player_vars:
                    for i_name in intermediate_vs_from_v[v]:
                        intermediate_v = g.vs.find(label=i_name)
                        if out_edge.source == out_edge.target:
                            g.add_edge(
                                source=intermediate_v,
                                target=source,
                                label="max"
                            )
                        else:
                            target = g.vs.find(label=lts.vs[out_edge.target]["label"][~0])
                            g.add_edge(
                                source=intermediate_v,
                                target=target,
                                label="max"
                            )
    return g


def plot_game(g: Graph, player_name: str):
    print(f"Creating plot for player: {player_name}")
    print(f"Graph has {g.vcount()} vertices and {g.ecount()} edges")

    # Set visual style
    visual_style = {}

    # Vertex styling
    vertex_colors = []
    vertex_shapes = []
    for vertex in g.vs:
        if vertex["type"] == "state":
            vertex_colors.append("lightblue")
            vertex_shapes.append("circle")
        else:
            vertex_colors.append("lightgreen")
            vertex_shapes.append("square")

    visual_style["vertex_color"] = vertex_colors
    visual_style["vertex_shape"] = vertex_shapes
    visual_style["vertex_size"] = 40
    visual_style["vertex_label"] = g.vs["label"]
    visual_style["vertex_label_size"] = 12

    edge_colors = []
    for edge in g.es:
        if edge["label"] == "min":
            edge_colors.append("red")
        else:
            edge_colors.append("blue")

    visual_style["edge_color"] = edge_colors
    visual_style["edge_width"] = 2
    visual_style["edge_label"] = g.es["label"]
    visual_style["edge_arrow_size"] = 1.0
    visual_style["edge_curved"] = 0.2

    # Layout
    visual_style["layout"] = g.layout("fr")
    visual_style["bbox"] = (1000, 800)
    visual_style["margin"] = 80

    # Save to file
    filename = f"game_{player_name}.png"
    plot(g, filename, **visual_style)
    print(f"Plot saved as: {filename}")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(filename)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Game for player: {player_name}')
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
        print(f"But the file was saved as {filename}")


def solve_game_with_meanpayoff(g: Graph, player_name: str):
    """Convert the game to meanpayoff format and solve it"""
    print(f"\n=== Solving game for player: {player_name} ===")

    # First, let's debug the game structure
    print(f"Game structure: {g.vcount()} vertices, {g.ecount()} edges")
    print("Vertices:")
    for v in g.vs:
        print(f"  {v['name']} (type: {v['type']}, payoff: {v['payoff']})")

    print("Edges:")
    for e in g.es:
        source_name = g.vs[e.source]['name']
        target_name = g.vs[e.target]['name']
        print(f"  {source_name} -> {target_name} (label: {e['label']})")

    # Create a temporary file for the game data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_filename = f.name

        # Convert the game to meanpayoff format
        edges_written = convert_game_to_meanpayoff_format(g, f)

    print(f"Created game file with {edges_written} edges: {temp_filename}")

    # Let's also print the contents of the file for debugging
    print("Game file contents:")
    with open(temp_filename, 'r') as f:
        print(f.read())

    try:
        # Call the meanpayoff solver with current working directory
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
        # Clean up the temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass


def convert_game_to_meanpayoff_format(g: Graph, file_handle):
    edges_written = 0

    vertex_mapping = {}
    for i, vertex in enumerate(g.vs):
        vertex_mapping[vertex['name']] = i

    print("Vertex mapping:")
    for name, idx in vertex_mapping.items():
        print(f"  {name} -> {idx}")

    written_edges = set()
    for edge in g.es:
        source_id = vertex_mapping[g.vs[edge.source]['name']]
        target_id = vertex_mapping[g.vs[edge.target]['name']]

        if source_id == target_id:
            print(f"  Skipping self-loop: {source_id} -> {target_id}")
            continue

        edge_key = (source_id, target_id)
        if edge_key in written_edges:
            print(f"  Skipping duplicate edge: {source_id} -> {target_id}")
            continue

        weight = calculate_edge_weight(g, edge)

        file_handle.write(f"{source_id} {target_id} {weight}\n")
        written_edges.add(edge_key)
        edges_written += 1
        print(f"  Writing edge: {source_id} -> {target_id} (weight: {weight})")

    all_states = set(vertex_mapping.values())
    states_with_edges = set(source for source, target in written_edges)
    states_without_edges = all_states - states_with_edges

    if states_without_edges:
        print(f"Warning: States without outgoing edges: {states_without_edges}")
        for state in states_without_edges:
            file_handle.write(f"{state} {state} 0\n")
            edges_written += 1
            print(f"  Adding self-loop for isolated state: {state} -> {state} (weight: 0)")

    return edges_written


def calculate_edge_weight(g: Graph, edge):
    pass


def parse_meanpayoff_results(output: str, player_name: str):
    """Parse the output from meanpayoff solver and extract useful information"""
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
