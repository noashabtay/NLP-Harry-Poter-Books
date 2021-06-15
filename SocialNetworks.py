import networkx as nx
import numpy
import matplotlib.pyplot as plt


def construct_network_from_neighbours_list(related_characters: list):
    graph = nx.Graph()
    for edge in related_characters:
        graph.add_edge(edge[0], edge[1])

    return graph


def draw_graph(graph: nx.Graph):
    nx.draw(graph, with_labels=True)
    plt.show()
