import math
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


# Generates the road network graph from the roads.csv file in the /data/ folder
def generate_graph():
    start = time.time()
    df = pd.read_csv(file_name)

    # Different groups of roads to run the analysis on/debug on
    all_roads = df["road"].unique().tolist()
    n_roads = [road for road in all_roads if road[0] == "N"]
    n12_roads = [road for road in all_roads if road[:2] == "N1"] + [
        road for road in all_roads if road[:2] == "N2"
    ]
    r_roads = [road for road in all_roads if road[0] == "R"]
    z_roads = [road for road in all_roads if road[0] == "Z"]
    graph = nx.Graph()

    # These are the penalty coefficients for rerouting trucks (mainly large trucks) on small roads such as Z and R roads. These are explained in the report
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

    # Loops over all the elements in the road from the roads.csv file
    for df in df_objects_all:
        for _, row in df.iterrows():  # index, row in ...
            model_type = row["model_type"].strip()

            name = row["name"]
            if pd.isna(name):
                name = ""
            else:
                name = name.strip()

            if model_type == "sourcesink":
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
                        road=row["road"],
                        ids=(current_edge_start["id"], row["id"]),
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
                    # Stores the traffic values on this bridge to be stored in the edges of the graphs
                    if row[truck_type] in current_edge_traffic[truck_type]:
                        current_edge_traffic[truck_type][row[truck_type]] += 1
                    else:
                        current_edge_traffic[truck_type][row[truck_type]] = 1

            elif model_type == "link":
                current_edge_length += row["length"]
                for truck_type in ["Heavy Truck", "Medium Truck", "Small Truck"]:
                    # Stores the traffic values on this link to be stored in the edges of the graphs
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
                        road=row["road"],
                        ids=(current_edge_start["id"], row["id"]),
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
    return graph


# Given a dict storing the amount of bridges of each condition, returns the probability of at least one of them failing (which means that the edge is shutdown)
def get_edge_shutdown_probability(dict):
    working_edge_probability = 1
    for i in dict:
        working_edge_probability *= (1 - ((1 / 10) ** (4 - i))) ** dict[i]
    return 1 - working_edge_probability


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

    # Approximates the number of tons that passes through the edge
    traffic = np.array(
        [
            25 * d.get("heavy_truck_traffic", 1)
            + 10 * d.get("medium_truck_traffic", 1)
            + 5 * d.get("small_truck_traffic", 1)
            for _, _, d in edges
        ]
    )

    traffic_safe = np.clip(traffic, 1e-3, None)
    log_t = np.log2(traffic_safe)
    log_min, log_max = np.nanmin(log_t), np.nanmax(log_t)
    traffic_scaled = (log_t - log_min) / (log_max - log_min + 1e-9)
    widths = 0.1 + traffic_scaled * 4

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


# Computes the economical impact of shutting down for every edge in the graph
def get_edges_criticality(graph):

    # Number of tons each type of truck can transport
    trucks_tonnage = {"Heavy Truck": 25, "Medium Truck": 10, "Small Truck": 5}

    edges_list = graph.edges()
    # We compute the metric for every edge in the graph
    edge_penalties = []
    indexes_of_blockages = []
    counter = 0
    for edge in edges_list:
        edge_attributes = graph.edges[edge]
        # After copying the edge attributes for safekeeping, we remove the edge to evaluate the alternative paths
        graph.remove_edge(*edge)
        edge_penalty = 0
        if nx.has_path(graph, edge[0], edge[1]):
            for truck_type, traffic_type, weight_type in [
                ("Heavy Truck", "heavy_truck_traffic", "heavy_truck_weight"),
                ("Medium Truck", "medium_truck_traffic", "medium_truck_weight"),
                ("Small Truck", "small_truck_traffic", "small_truck_weight"),
            ]:
                # We compute the shortest path, taking into account the time cost of rerouting big trucks into small roads
                shortest_path = nx.shortest_path(
                    graph, edge[0], edge[1], weight=weight_type
                )
                path_length = 0
                for i in range(len(shortest_path) - 1):
                    path_length += graph.edges[
                        (shortest_path[i], shortest_path[i + 1])
                    ][weight_type]
                detour = path_length - edge_attributes["length"]
                edge_penalty += (
                    detour
                    * edge_attributes[traffic_type]
                    * trucks_tonnage[truck_type]
                    / 48
                )
            edge_penalties.append((edge_penalty, edge_attributes))
        else:
            indexes_of_blockages.append(counter)
            counter += 1
            tonnage_blocked = 0
            for truck_type, traffic_type, weight_type in [
                ("Heavy Truck", "heavy_truck_traffic", "heavy_truck_weight"),
                ("Medium Truck", "medium_truck_traffic", "medium_truck_weight"),
                ("Small Truck", "small_truck_traffic", "small_truck_weight"),
            ]:
                if not math.isnan(edge_attributes[traffic_type]):
                    tonnage_blocked += (
                        edge_attributes[traffic_type] * trucks_tonnage[truck_type]
                    )
            edge_penalties.append((tonnage_blocked, edge_attributes))

        # We add teh edge back to continue the graph analysis on another one
        graph.add_edge(edge[0], edge[1], **edge_attributes)
    return edge_penalties, indexes_of_blockages


# Return the expected value of the delays (in tons*hours) by multiplying the shutdown probability and the delay of goods it would cause
def get_expected_tonnage_delay(graph, delta_time_hours):
    penalties, indexes = get_edges_criticality(graph)
    for i in range(len(penalties)):
        if i in indexes:
            penalties[i] = (
                penalties[i][0]
                * delta_time_hours
                * penalties[i][1]["shutdown_probability"],
                penalties[i][1],
            )
        else:
            penalties[i] = (
                penalties[i][0] * penalties[i][1]["shutdown_probability"],
                penalties[i][1],
            )
    return penalties


# EOF -----------------------------------------------------------
