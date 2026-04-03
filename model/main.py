import numpy as np

import model

print("Building graph...")
graph = model.generate_graph()
print("Graph built")
if bool(input("Do you want to display the graph ? (leave empty if not) : ")):
    model.draw_graph(graph)

crisis_length = float(input("For how long does the event last ? (in hours) : "))
penalties = model.get_expected_tonnage_delay(graph, crisis_length)
penalties.sort(key=lambda tup: tup[0], reverse=True)
for i in range(10):
    element = penalties[i]
    print(
        f"{i + 1} - {int(element[0])} Tons*hours of expected delay on road {element[1]['road']} between ids {element[1]['ids']} ({element[1]['shutdown_probability']})"
    )
