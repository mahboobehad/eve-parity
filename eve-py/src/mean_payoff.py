import itertools

from igraph import Graph, plot

import parsrml as game_spec


def create_zero_sum_games(lts: Graph):
    players = [(list(m[1])[0], m[2]) for m in game_spec.modules]

    zero_sum_turn_based_games = []
    for name, owned_vars in players:
        g = create_player_game(lts, name, owned_vars)
        zero_sum_turn_based_games.append(g)

        plot_game(g, name)


def create_player_game(lts: Graph, player_name: str, player_vars: set):
    g = Graph(directed=True)

    for v in lts.vs:
        v_payoff = None
        for payoff_spec in game_spec.payoffs:
            player = payoff_spec["module"]
            value = payoff_spec["value"]
            state = payoff_spec["state"]
            if player == player_name and state in v["label"][~0]:
                v_payoff = value
                break

        g.add_vertex(name=f"v{v.index}", label=v["label"][~0], type="state", payoff=v_payoff)
        source = g.vs.find(name=f"v{v.index}")

        for edge_index in lts.incident(v, mode="out"):
            out_edge = lts.es[edge_index]
            if not out_edge["direction"]:
                continue

            coalition_actions_from_v = set()
            intermediate_vs_from_v = set()
            for var in out_edge["direction"]:
                if var and var not in player_vars:
                    coalition_actions_from_v.add(var)

            if coalition_actions_from_v:
                actions_list = sorted(list(coalition_actions_from_v))

                for combo in itertools.product([True, False], repeat=len(actions_list)):
                    combo_parts = []
                    for i, action in enumerate(actions_list):
                        combo_parts.append(f"{action}_{str(combo[i]).lower()}")
                    intermediate_name = "_".join(combo_parts)
                    full_node_name = f"v{v.index}_{intermediate_name}"
                    try:
                        existing_node = g.vs.find(name=full_node_name)
                        intermediate_vertex = existing_node
                    except ValueError:
                        g.add_vertex(name=full_node_name,
                                     label=intermediate_name,
                                     type="intermediate",
                                     payoff=v_payoff)
                        intermediate_vs_from_v.add(full_node_name)
                        intermediate_vertex = g.vs.find(name=full_node_name)

                    g.add_edge(source=source,
                               target=intermediate_vertex,
                               label="min")

            for var in out_edge["direction"]:
                if var and var in player_vars:
                    for i_name in intermediate_vs_from_v:
                        intermediate_v = g.vs.find(name=i_name)
                        if out_edge.source == out_edge.target:
                            g.add_edge(source=intermediate_v,
                                       target=v,
                                       label="max")
                        else:
                            g.add_edge(source=intermediate_v,
                                       target=out_edge.target,
                                       label="max")

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
