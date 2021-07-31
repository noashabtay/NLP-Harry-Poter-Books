import networkx as nx
import pickle
import matplotlib.pyplot as plt


def construct_network_from_neighbours_list(related_characters: list):
    """
    Gets a list of the related characters (edges), the sentiment of their relation and the amount of common appearances,
    and creates a graph with the edges, their colors (determined by the sentiment) and their weights (determined by
    their common appearances).
    """
    graph = nx.Graph()
    for edge in related_characters:
        sentiment = edge[1]
        color = ''
        if sentiment == 'Positive':
            color = 'g'
        elif sentiment == 'Negative':
            color = 'r'
        elif sentiment == 'Neutral':
            color = 'k'
        # graph.add_node(edge[0][0], popularity=
        graph.add_edge(edge[0][0], edge[0][1], color=color, weight=edge[2])

    return graph


def draw_graph(graph: nx.Graph):
    """
    Draws the graph.
    """
    colors = [graph[u][v]['color'] for u, v in graph.edges()]
    nx.draw(graph, nx.spring_layout(graph), edge_color=colors, with_labels=True)

    plt.show()


def compute_centrality_for_nodes(graph: nx.Graph):
    """
    Computes the degree, closeness and betweeness centrality of the nodes in the graph, and creates a dictionary with
    the nodes as keys and their centralities as values.
    """
    nodes_centralities = {}
    degree_centralities = nx.degree_centrality(graph)
    betweeness_centralities = nx.betweenness_centrality(graph, normalized=True)
    closeness_centralities = nx.closeness_centrality(graph)
    for node in graph.nodes:
        closeness = closeness_centralities[node]
        degree = degree_centralities[node]
        betweeness = betweeness_centralities[node]
        nodes_centralities[node] = {
            "degree": degree,
            "closeness": closeness,
            "betweeness": betweeness
        }

    return nodes_centralities


def load_all_graphs():
    """
    Reads all the gml files of the graphs and stores them in a list.
    Each index corresponds to the book number in the series.
    """
    all_graphs = []
    for i in range(7):
        with open(f'Full_Network_Book_{i+1}.gml', 'rb') as graph_file:
            all_graphs.append(nx.read_gml(graph_file))

    return all_graphs


