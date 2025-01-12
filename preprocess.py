import networkx
import os


def convert_all_graphml_to_edgelist(path):
    for file in os.listdir(path):
        if file.endswith(".graphml"):
            G = networkx.read_graphml(os.path.join(path, file))
            G = networkx.convert_node_labels_to_integers(
                G, ordering="sorted"
            )
            # generate edgelist
            networkx.write_edgelist(G, os.path.join(
                path, file[:-8] + ".edgelist"), data=False)
            os.remove(os.path.join(path, file))
        else:
            # graph end with ".edgelist"
            G = networkx.read_edgelist(os.path.join(path, file))
            networkx.write_edgelist(G, os.path.join(
                path, file), data=False)


def preprocess_data():
    print("preprocess : train")
    path = "data/data/train/graph"  # o
    convert_all_graphml_to_edgelist(path)
    print("preprocess : valid")
    path = "data/data/valid/graph"
    convert_all_graphml_to_edgelist(path)


preprocess_data()
