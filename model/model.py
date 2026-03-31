import os
import time
from operator import is_not

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
    r_roads = [road for road in all_roads if road[0] == "R"]
    z_roads = [road for road in all_roads if road[0] == "Z"]
    graph = nx.Graph()

    reroute_penalty_coefficients = {
        "N": {"Heavy Truck": 1, "Medium Truck": 1, "Small Truck": 1},
        "R": {"Heavy Truck": 5, "Medium Truck": 1, "Small Truck": 1},
        "Z": {"Heavy Truck": 50, "Medium Truck": 1.25, "Small Truck": 1},
    }

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
    current_edge_length = 0
    current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}
    current_edge_traffic = {
        "Heavy Truck": {0: 0},
        "Medium Truck": {0: 0},
        "Small Truck": {0: 0},
    }

    for df in df_objects_all:
        for _, row in df.iterrows():  # index, row in ...
            # create agents according to model_type
            model_type = row["model_type"].strip()

            name = row["name"]
            traffic_weight = (
                25 * row["Heavy Truck"]
                + 10 * row["Medium Truck"]
                + 5 * row["Small Truck"]
            )
            traffic_weight = row["Small Truck"]
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
                    heavy_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Heavy Truck"]
                        * current_edge_length
                    )
                    medium_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Medium Truck"]
                        * current_edge_length
                    )
                    small_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Small Truck"]
                        * current_edge_length
                    )
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        length=current_edge_length,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                        heavy_truck_traffic=max(
                            current_edge_traffic["Heavy Truck"],
                            key=current_edge_traffic["Heavy Truck"].get,
                        ),
                        medium_truck_traffic=max(
                            current_edge_traffic["Medium Truck"],
                            key=current_edge_traffic["Medium Truck"].get,
                        ),
                        small_truck_traffic=max(
                            current_edge_traffic["Small Truck"],
                            key=current_edge_traffic["Small Truck"].get,
                        ),
                        heavy_truck_weight=heavy_truck_weight,
                        medium_truck_weight=medium_truck_weight,
                        small_truck_weight=small_truck_weight,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_length = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}
                current_edge_traffic = {
                    "Heavy Truck": {0: 0},
                    "Medium Truck": {0: 0},
                    "Small Truck": {0: 0},
                }

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
                    heavy_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Heavy Truck"]
                        * current_edge_length
                    )
                    medium_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Medium Truck"]
                        * current_edge_length
                    )
                    small_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Small Truck"]
                        * current_edge_length
                    )
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        length=current_edge_length,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                        heavy_truck_traffic=max(
                            current_edge_traffic["Heavy Truck"],
                            key=current_edge_traffic["Heavy Truck"].get,
                        ),
                        medium_truck_traffic=max(
                            current_edge_traffic["Medium Truck"],
                            key=current_edge_traffic["Medium Truck"].get,
                        ),
                        small_truck_traffic=max(
                            current_edge_traffic["Small Truck"],
                            key=current_edge_traffic["Small Truck"].get,
                        ),
                        heavy_truck_weight=heavy_truck_weight,
                        medium_truck_weight=medium_truck_weight,
                        small_truck_weight=small_truck_weight,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_length = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}
                current_edge_traffic = {
                    "Heavy Truck": {0: 0},
                    "Medium Truck": {0: 0},
                    "Small Truck": {0: 0},
                }

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
                    heavy_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Heavy Truck"]
                        * current_edge_length
                    )
                    medium_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Medium Truck"]
                        * current_edge_length
                    )
                    small_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Small Truck"]
                        * current_edge_length
                    )
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        length=current_edge_length,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                        heavy_truck_traffic=max(
                            current_edge_traffic["Heavy Truck"],
                            key=current_edge_traffic["Heavy Truck"].get,
                        ),
                        medium_truck_traffic=max(
                            current_edge_traffic["Medium Truck"],
                            key=current_edge_traffic["Medium Truck"].get,
                        ),
                        small_truck_traffic=max(
                            current_edge_traffic["Small Truck"],
                            key=current_edge_traffic["Small Truck"].get,
                        ),
                        heavy_truck_weight=heavy_truck_weight,
                        medium_truck_weight=medium_truck_weight,
                        small_truck_weight=small_truck_weight,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_length = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}
                current_edge_traffic = {
                    "Heavy Truck": {0: 0},
                    "Medium Truck": {0: 0},
                    "Small Truck": {0: 0},
                }

            elif model_type == "bridge":
                current_edge_length += row["length"]
                current_edge_bridge_conditions[row["condition"]] += 1
                for truck_type in ["Heavy Truck", "Medium Truck", "Small Truck"]:
                    if row[truck_type] in current_edge_traffic[truck_type]:
                        current_edge_traffic[truck_type][row[truck_type]] += 1
                    else:
                        current_edge_traffic[truck_type][row[truck_type]] = 1

            elif model_type == "link":
                current_edge_length += row["length"]
                for truck_type in ["Heavy Truck", "Medium Truck", "Small Truck"]:
                    if row[truck_type] in current_edge_traffic[truck_type]:
                        current_edge_traffic[truck_type][row[truck_type]] += 1
                    else:
                        current_edge_traffic[truck_type][row[truck_type]] = 1

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
                    heavy_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Heavy Truck"]
                        * current_edge_length
                    )
                    medium_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Medium Truck"]
                        * current_edge_length
                    )
                    small_truck_weight = (
                        reroute_penalty_coefficients[road[0]]["Small Truck"]
                        * current_edge_length
                    )
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        length=current_edge_length,
                        shutdown_probability=get_edge_shutdown_probability(
                            current_edge_bridge_conditions
                        ),
                        heavy_truck_traffic=max(
                            current_edge_traffic["Heavy Truck"],
                            key=current_edge_traffic["Heavy Truck"].get,
                        ),
                        medium_truck_traffic=max(
                            current_edge_traffic["Medium Truck"],
                            key=current_edge_traffic["Medium Truck"].get,
                        ),
                        small_truck_traffic=max(
                            current_edge_traffic["Small Truck"],
                            key=current_edge_traffic["Small Truck"].get,
                        ),
                        heavy_truck_weight=heavy_truck_weight,
                        medium_truck_weight=medium_truck_weight,
                        small_truck_weight=small_truck_weight,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_length = 0
                current_edge_bridge_conditions = {0: 0, 1: 0, 2: 0, 3: 0}
                current_edge_traffic = {
                    "Heavy Truck": {0: 0},
                    "Medium Truck": {0: 0},
                    "Small Truck": {0: 0},
                }

    print("elapsed time :", time.time() - start, "seconds")
    # graph = prune_graph(graph, "length")
    # print(
    #     list(nx.connected_components(graph)), len(list(nx.connected_components(graph)))
    # )
    return graph


def get_edge_shutdown_probability(dict):
    working_edge_probability = 1
    for i in dict:
        working_edge_probability *= (1 - ((1 / 10) ** (4 - i))) ** dict[i]
    return 1 - working_edge_probability


# Remove nodes of degree 2 for increased clarity
def prune_graph(graph, weight_attr="length", prob_attr="shutdown_probability"):
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
    # Positions
    pos = {n: (d["lon"], d["lat"]) for n, d in graph.nodes(data=True)}

    # Node styling
    colors, sizes = [], []
    for n, d in graph.nodes(data=True):
        roads = d.get("road", [])
        road_types = [road[0] for road in roads]
        if False and not nx.has_path(graph, n, 1000000):
            colors.append("red")
            sizes.append(50)
        else:
            if "N" in road_types:
                colors.append("black")
                sizes.append(20)
            elif "R" in road_types:
                colors.append("black")
                sizes.append(5)
            else:
                colors.append("black")
                sizes.append(2)

    # Edge data
    edges = list(graph.edges(data=True))
    probs = np.array([d.get("shutdown_probability", 0) for _, _, d in edges])

    traffic = np.array([d.get("heavy_truck_traffic", 1) for _, _, d in edges])
    traffic_safe = np.clip(traffic, 1e-3, None)
    log_t = np.log2(traffic_safe)
    log_min, log_max = np.nanmin(log_t), np.nanmax(log_t)
    traffic_scaled = (log_t - log_min) / (log_max - log_min + 1e-9)
    widths = 0.1 + traffic_scaled * 4

    # plt.hist(
    #     [np.log10(prob) for prob in probs if prob > 0],
    #     color="green",
    #     edgecolor="black",
    #     bins=10,
    # )
    # plt.title("Histogram of shutdown probabilities")
    # plt.show()
    # print([float(np.log10(prob)) for prob in probs])

    # --- LOG SCALING ---
    eps = 1e-3
    probs_safe = np.clip(probs, eps, 1)

    log_probs = np.log10(probs_safe)

    # Normalize log values to [0, 1]
    log_min = np.log10(eps)
    log_max = 0  # log10(1)
    probs_scaled = (log_probs - log_min) / (log_max - log_min)

    # --- DRAW ---
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color=colors, ax=ax)

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        edge_color=probs_scaled,
        edge_cmap=plt.cm.plasma,  # green → red
        edge_vmin=0,
        edge_vmax=1,
        width=widths,
        alpha=0.9,
        arrows=False,
        ax=ax,
    )

    # --- COLORBAR (log scale labels!) ---
    norm = mcolors.LogNorm(vmin=eps, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Shutdown Probability of Edges (log scale)")

    # GIS-style cleanup
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()

    plt.show()
    return


# Computes the economical impact of breaking down for every edge in the graph
def get_edges_criticality(graph):
    edges_list = graph.edges()
    alternative_paths = {}
    for edge in edges_list:
        edge_attributes = graph.edges[edge]
        graph.remove_edge(*edge)
        for weight_type in [
            "heavy_truck_weight",
            "medium_truck_weight",
            "small_truck_weight",
        ]:
            if nx.has_path(graph, edge[0], edge[1]):
                shortest_path = nx.shortest_path(
                    graph, edge[0], edge[1], weight="length"
                )
                path_length = 0
                for i in range(len(shortest_path) - 1):
                    path_length += graph.edges[
                        (shortest_path[i], shortest_path[i + 1])
                    ]["length"]
                    alternative_paths[edge] = path_length - edge_attributes["length"]
                print(path_length, edge_attributes["length"], edge)
        else:
            alternative_paths[edge] = None
        graph.add_edge(edge[0], edge[1], **edge_attributes)
    # print(
    #     sum(v for v in alternative_paths.values() if v is not None)
    #     / sum(1 for v in alternative_paths.values() if v is not None)
    # )
    return


graph = generate_graph()
# get_edges_criticality(graph)
draw_graph(graph)
# EOF -----------------------------------------------------------
