# convert pred/**.txt edgelist to csv

import os
import networkx
import csv
import re


pred_dir = "pred"
csv_dir = "csv"
os.makedirs(csv_dir, exist_ok=True)
with open("output_sft.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    # order the files
    # read test.txt
    with open("data/test/test.txt", "r") as f:
        test = f.readlines()

    def extract_numbers(text):
        # Use regular expression to find integers and floats
        numbers = re.findall(r'\d+\.\d+|\d+', text)
        # Convert the extracted numbers to float
        return [float(num) for num in numbers]

    stats_graph = []
    for line in test:
        stats = extract_numbers(line)
        stats_graph.append(stats)

    for filename in range(1000):
        edgelist_path = os.path.join(pred_dir, f"{filename}.txt")
        graph_id = f"graph_{str(filename)}"

        G = networkx.read_edgelist(edgelist_path)
        # Define a graph ID
        # normalize the graph
        G = networkx.convert_node_labels_to_integers(G)
        # if nb of nodes is >= 50
        if G.number_of_nodes() > 50:
            # cut the graph
            G = G.subgraph(list(G.nodes())[:50])

        # plot graph
        networkx.draw(G, with_labels=True)
        # print properties
        # print(f"Graph ID: {graph_id}")
        # print(f"Number of nodes: {G.number_of_nodes()} : {
        #       stats_graph[filename][1]}")
        # print(f"Number of edges: {G.number_of_edges()} : {
        #       stats_graph[filename][2]}")
        # # avg degree
        # print(f"Average degree: {sum(dict(G.degree()).values(
        # ))/G.number_of_nodes():2f} : {stats_graph[filename][3]:2f}")
        # # nb of triangles
        # print(f"Number of triangles: {sum(networkx.triangles(G).values())} : {
        #       stats_graph[filename][4]}")
        # # global clustering coefficient
        # print(f"Global clustering coefficient: {
        #       networkx.transitivity(G)} : {stats_graph[filename][5]}")
        # # community number
        # print(f"Number of communities: {networkx.number_connected_components(G)} : {
        #       stats_graph[filename][7]}")

        edge_list_text = ", ".join(
            [f"({u}, {v})" for u, v in G.edges()])
        # Write the graph ID and the full edge list as a single row
        writer.writerow([graph_id, edge_list_text])
