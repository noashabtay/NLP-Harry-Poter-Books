import networkx as nx
import pickle
import matplotlib.pyplot as plt


def construct_network_from_neighbours_list(related_characters: list):
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
    # nx.draw(graph, with_labels=True)
    # nx.draw_kamada_kawai(graph, with_labels=True)
    # edge_labels = nx.get_edge_attributes(graph, 'sentiment')
    colors = [graph[u][v]['color'] for u, v in graph.edges()]
    nx.draw(graph, nx.spring_layout(graph), edge_color=colors, with_labels=True)
    # nx.draw_circular(graph, with_labels=True)

    plt.show()


with open('/Users/noashabtay/PycharmProjects/NLP/related_characters_book_1.pkl', 'rb') as rel_characters_book_1:
    rel_characters_book_1 = pickle.load(rel_characters_book_1)

    graph = construct_network_from_neighbours_list(rel_characters_book_1)
    nx.write_gml(graph, "/Users/noashabtay/PycharmProjects/NLP/network_full.gml")
    # draw_graph(graph)
