import os
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_name = os.path.join(BASE_DIR, "data", "roads.csv")
speed = 48 * 1000 / 60


def generate_graph():
    start = time.time()
    """
    generate the simulation model according to the csv file component information

    Warning: the labels are the same as the csv column labels
    """
    df = pd.read_csv(file_name)

    # a list of names of roads to be generated
    all_roads = df["road"].unique().tolist()
    n_roads = [road for road in all_roads if road[0] == "N"]
    n12_roads = [road for road in all_roads if road[:2] == "N1"] + [
        road for road in all_roads if road[:2] == "N2"
    ]
    graph = nx.Graph()

    df_objects_all = []
    for road in all_roads:
        # Select all the objects on a particular road in the original order as in the cvs
        df_objects_on_road = df[df["road"] == road]

        if not df_objects_on_road.empty:
            df_objects_all.append(df_objects_on_road)

    # put back to df with selected roads so that min and max and be easily calculated
    df = pd.concat(df_objects_all)

    # Store the information about the start of the segment of road the program is currently following
    current_edge_start = {"road": None, "id": None}
    current_edge_weight = 0
    current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}

    for df in df_objects_all:
        for _, row in df.iterrows():  # index, row in ...
            # create agents according to model_type
            model_type = row["model_type"].strip()

            name = row["name"]
            if pd.isna(name):
                name = ""
            else:
                name = name.strip()

            if model_type == "source":
                # We add a node corresponding to the element and link it to the previous node if they are on the same road
                graph.add_node(
                    row["id"],
                    road={row["road"]},
                    type=model_type,
                    lat=row["lat"],
                    lon=row["lon"],
                )
                if current_edge_start["road"] == row["road"]:
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        weight=current_edge_weight,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}

            elif model_type == "sink":
                # We add a node corresponding to the element and link it to the previous node if they are on the same road
                graph.add_node(
                    row["id"],
                    road={row["road"]},
                    type=model_type,
                    lat=row["lat"],
                    lon=row["lon"],
                )
                if current_edge_start["road"] == row["road"]:
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        weight=current_edge_weight,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}

            elif model_type == "sourcesink":
                # We add a node corresponding to the element and link it to the previous node if they are on the same road
                graph.add_node(
                    row["id"],
                    road={row["road"]},
                    type=model_type,
                    lat=row["lat"],
                    lon=row["lon"],
                )
                if current_edge_start["road"] == row["road"]:
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        weight=current_edge_weight,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}

            elif model_type == "bridge":
                current_edge_weight += row["length"]
                current_edge_bridge_conditions[row["condition"]] += 1

            elif model_type == "link":
                current_edge_weight += row["length"]

            elif model_type == "intersection":
                # Intersection elements are stored in multiple roads, it is necessary to check wether it is already in the graph or not.
                if row["id"] not in list(graph.nodes):
                    graph.add_node(
                        row["id"],
                        road={row["road"]},
                        type=model_type,
                        lat=row["lat"],
                        lon=row["lon"],
                    )
                else:  # if the intersection has already been added from another road
                    graph.nodes[row["id"]]["road"].add(row["road"])

                # We add a node corresponding to the element and link it to the previous node if they are on the same road
                if current_edge_start["road"] == row["road"]:
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        weight=current_edge_weight,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}

    print("delta", time.time() - start)
    # graph = prune_graph(graph, "weight")
    draw_graph(graph)
    return graph


def get_edge_shutdown_probability(dict):
    working_edge_probability = 1
    for i in dict:
        working_edge_probability *= (1 - ((1 / 10) ** (4 - i))) ** dict[i]
    return 1 - working_edge_probability


def prune_graph(graph, weight_attr="weight", prob_attr="shutdown_probability"):
    # We iterate until no more degree-2 nodes exist
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes()):
            if graph.degree(node) != 2:
                continue

            neighbors = list(graph.neighbors(node))
            u, v = neighbors

            # graphet weights (default = 1 if missing)
            w1 = graph[node][u].get(weight_attr, 1)
            w2 = graph[node][v].get(weight_attr, 1)
            new_weight = w1 + w2

            p1 = graph[node][u].get(prob_attr, 1)
            p2 = graph[node][v].get(prob_attr, 1)
            new_prob = 1 - ((1 - p1) * (1 - p2))

            # If edge already exists, sum weights
            if graph.has_edge(u, v):
                existing_weight = graph[u][v].get(weight_attr, 1)
                graph[u][v][weight_attr] = existing_weight + new_weight
            else:
                graph.add_edge(u, v, **{weight_attr: new_weight, prob_attr: new_prob})

            graph.remove_node(node)
            changed = True
            break  # restart iteration (graph changed)
    return graph


# draws the graph of the road network
def draw_graph(graph):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Positions
    pos = {n: (d["lon"], d["lat"]) for n, d in graph.nodes(data=True)}

    # Node styling
    colors, sizes = [], []
    for _, d in graph.nodes(data=True):
        roads = d.get("road", [])
        road_types = [road[0] for road in roads]

        if "N" in road_types:
            colors.append("purple")
            sizes.append(20)
        elif "R" in road_types:
            colors.append("purple")
            sizes.append(10)
        else:
            colors.append("purple")
            sizes.append(5)

    # Edge data
    edges = list(graph.edges(data=True))
    probs = np.array([d.get("shutdown_probability", 0) for _, _, d in edges])
    print([float(np.log10(prob)) for prob in probs])

    # --- LOG SCALING ---
    eps = 1e-3
    probs_safe = np.clip(probs, eps, 1)

    log_probs = np.log10(probs_safe)

    # Normalize log values to [0, 1]
    log_min = np.log10(eps)
    log_max = 0  # log10(1)
    probs_scaled = (log_probs - log_min) / (log_max - log_min)

    # --- DRAW ---
    nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color=colors, ax=ax)

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        edge_color=probs_scaled,
        edge_cmap=plt.cm.RdYlGn_r,  # green → red
        edge_vmin=0,
        edge_vmax=1,
        width=1.5,
        alpha=0.9,
        arrows=False,
        ax=ax,
    )

    # --- COLORBAR (log scale labels!) ---
    norm = mcolors.LogNorm(vmin=eps, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Shutdown Probability of Edges (log scale)")

    # GIS-style cleanup
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()

    plt.show()
    return


# graphiven a source and a sink, sets the shortest (directed!) path between the two in the path_ids_dict as a list of ids
def update_path_dict(graph, source, sink):
    if not nx.has_path(graph, source, sink):
        print("ALLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERT", source, sink)
    nodes_list = nx.shortest_path(graph, source=source, target=sink, weight="weight")
    path = []
    route_mean_delay, length = 0, 0
    for i in range(len(nodes_list) - 1):
        path.append(nodes_list[i])
        path += graph[nodes_list[i]][nodes_list[i + 1]]["ids"]
        length += graph[nodes_list[i]][nodes_list[i + 1]]["weight"]
        route_mean_delay += graph[nodes_list[i]][nodes_list[i + 1]]["mean_delay"]
    path.append(nodes_list[-1])
    # path_ids_dict[source, sink] = (path, length, route_mean_delay)
    return


# gets a random route by choosing a random sink
# def get_random_route(graph, source):
#     """
#     pick up a random route given an origin
#     """
#     while True:
#         # different source and sink
#         sink = random.choice(sinks)
#         if sink is not source and nx.has_path(graph, source, sink):
#             break
#     # Ensures that each path is calculated at most once
#     if (source, sink) not in path_ids_dict:
#         update_path_dict(source, sink)

#     return path_ids_dict[source, sink][0]


# check if the graph is connected (for debugging)
# def check_is_graph_connected(graph):
#     n1_start = 1000000
#     for source in sources:
#         if not nx.has_path(graph, source, n1_start):
#             print(
#                 f"The {graph.nodes[source].get('road')} is not connected to the N1 road"
#             )
#             continue


# computes the shortest path for all the pairs of roads extremities
# def get_all_routes(graph):
#     for source in sources:
#         for sink in sinks:
#             if (sink is not source) and nx.has_path(graph, source, sink):
#                 update_path_dict(source, sink)
#     return path_ids_dict


# uses the random route choice
# def get_route(graph, source):
#     return get_random_route(graph, source)

graph = generate_graph()
# EOF -----------------------------------------------------------
