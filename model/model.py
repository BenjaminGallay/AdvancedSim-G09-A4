import os

import analytical_recorder
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from model import analytical_recorder

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_name = os.path.join(BASE_DIR, "data", "roads.csv")
speed = 48 * 1000 / 60


def generate_graph():
    """
    generate the simulation model according to the csv file component information

    Warning: the labels are the same as the csv column labels
    """
    df = pd.read_csv(file_name)

    # a list of names of roads to be generated
    # TODO You can also read in the road column to generate this list automatically
    roads = [
        "N1",
        "N101",
        "N102",
        "N103",
        "N104",
        "N105",
        "N106",
        "N107",
        "N108",
        "N109",
        "N110",
        "N111",
        "N112",
        "N119",
        "N120",
        "N123",
        "N128",
        "N129",
        "N2",
        "N203",
        "N204",
        "N205",
        "N206",
        "N207",
        "N208",
        "N209",
        "N210",
    ]

    all_roads = df["road"].unique().tolist()
    n_roads = [road for road in all_roads if road[0] == "N"]
    # roads = ["N1", "N102", "N103"]
    speed = 48 * 1000 / 60
    graph = nx.DiGraph()
    breakdown_probabilities = [0.05, 0.1, 0.2, 0.4]

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
    current_edge_id_list = []
    current_edge_mean_bridge_delay = 0

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
                    road=row["road"],
                    type=model_type,
                    lat=row["lat"],
                    lon=row["lon"],
                )
                if current_edge_start["road"] == row["road"]:
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        weight=current_edge_weight,
                        ids=current_edge_id_list,
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                    graph.add_edge(
                        row["id"],
                        current_edge_start["id"],
                        weight=current_edge_weight,
                        ids=current_edge_id_list[::-1],
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_id_list = []
                current_edge_mean_bridge_delay = 0

            elif model_type == "sink":
                # We add a node corresponding to the element and link it to the previous node if they are on the same road
                graph.add_node(
                    row["id"],
                    road=row["road"],
                    type=model_type,
                    lat=row["lat"],
                    lon=row["lon"],
                )
                if current_edge_start["road"] == row["road"]:
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        weight=current_edge_weight,
                        ids=current_edge_id_list,
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                    graph.add_edge(
                        row["id"],
                        current_edge_start["id"],
                        weight=current_edge_weight,
                        ids=current_edge_id_list[::-1],
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_id_list = []
                current_edge_mean_bridge_delay = 0

            elif model_type == "sourcesink":
                # We add a node corresponding to the element and link it to the previous node if they are on the same road
                graph.add_node(
                    row["id"],
                    road=row["road"],
                    type=model_type,
                    lat=row["lat"],
                    lon=row["lon"],
                )
                if current_edge_start["road"] == row["road"]:
                    graph.add_edge(
                        current_edge_start["id"],
                        row["id"],
                        weight=current_edge_weight,
                        ids=current_edge_id_list,
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                    graph.add_edge(
                        row["id"],
                        current_edge_start["id"],
                        weight=current_edge_weight,
                        ids=current_edge_id_list[::-1],
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_id_list = []
                current_edge_mean_bridge_delay = 0

            elif model_type == "bridge":
                current_edge_weight += row["length"]
                current_edge_id_list.append(row["id"])
                current_edge_mean_bridge_delay += (
                    analytical_recorder.compute_bridge_mean_delay(row["length"])
                ) * breakdown_probabilities[int(row["condition"])]

            elif model_type == "link":
                current_edge_weight += row["length"]
                current_edge_id_list.append(row["id"])

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
                        ids=current_edge_id_list,
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                    graph.add_edge(
                        row["id"],
                        current_edge_start["id"],
                        weight=current_edge_weight,
                        ids=current_edge_id_list[::-1],
                        mean_delay=current_edge_mean_bridge_delay,
                    )
                current_edge_start = {"road": row["road"], "id": row["id"]}
                current_edge_weight = 0
                current_edge_id_list = []
                current_edge_mean_bridge_delay = 0
    return graph


# draws the graph of the road network
def draw_graph(graph):
    pos = {n: (d["lon"], d["lat"]) for n, d in graph.nodes(data=True)}
    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color="purple")
    nx.draw_networkx_edges(graph, pos, edge_color="turquoise", arrows=False)
    plt.show()
    # check_is_graph_connected()
    return


# Given a source and a sink, sets the shortest (directed!) path between the two in the path_ids_dict as a list of ids
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
draw_graph(graph)
# EOF -----------------------------------------------------------
