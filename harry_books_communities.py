import networkx as nx
from networkx.algorithms import community
from community import community_louvain
from networkx.algorithms.community import k_clique_communities


def edge_selector_optimizer(network):
    betweeness = nx.algorithms.centrality.edge_betweenness_centrality(network, weight='weight')
    centrality = {edge: centrality / max(betweeness.values())
                                           for edge, centrality in betweeness.items()}
    return max(centrality, key=betweeness.get)


def partition_modularity_calc(part, network):
    """
    calculate modularity of single partition
    :param part: single partition - list of nodes.
    :param network:
    :return:
    """
    sub_graph = network.subgraph(part)
    lc = sub_graph.number_of_edges()
    l = network.number_of_edges()
    kc = sum([val for (node, val) in sub_graph.degree()])
    first_part = lc / l
    second_part = (kc/(2*l))**2

    return first_part - second_part



def clique_percolation(network):
    cliques = list(nx.algorithms.clique.find_cliques(network))

    cliques_size = [len(clique) for clique in cliques]
    min_clique = min(cliques_size)
    max_clique = max(cliques_size)

    max_modularity = None
    best_partition = None

    for k in range(2, max_clique):
        partition = list(k_clique_communities(network, k))
        modularity_total = 0
        for part in partition:
            modularity_total += partition_modularity_calc(part, network)
        if max_modularity is None:
            max_modularity = modularity_total
            best_partition = partition
        elif max_modularity < modularity_total:
            max_modularity = modularity_total
            best_partition = partition

    return best_partition, max_modularity


def change_to_key_value(best_partition):
    best_partition_new = {}
    i = 0
    for part in best_partition:
        for node in part:
            best_partition_new[node] = i
        i += 1
    return best_partition_new


def community_detector(algorithm_name, network, most_valualble_edge=None):
    """
    :param algorithm_name: String - Name of the algorithm to run. Can be either
                                    ‘girvin_newman’, ‘louvain’ or ‘clique_percolation’
    :param network: networkX object - The network to run the detection over
    :param most_valualble_edge: function (or None) - A parameter that is used only by the ‘girvin_newman’ algorithm
    :return: dictionary { "num_partitions": int - Number of divided_books_graphs the network was divided to
                          "modularity": float - The modularity value of the partition
                          "partition": List of lists - The partition of the network.
                                       Each element in the list is a community detected (with node names)
    """
    if network.is_directed():
        network = network.to_undirected()
    result = {}
    partition = []  # list of lists - each community holds the key values
    modularity_val = 0

    if algorithm_name == 'girvin_newman':
        communities_generator = community.girvan_newman(network, most_valuable_edge=most_valualble_edge)
        communities_generator_list = list(communities_generator)
        # find the best partition:
        best_modul = None
        best_partition = None
        i = 0
        # for each partition -calc modularity and take the max
        for communities in communities_generator_list:
            modularity_val = 0
            j = 0
            for part in communities:
                modularity_val += partition_modularity_calc(part, network)
                j += 1
            if best_modul is None:
                best_modul = modularity_val
                best_partition = communities
            elif best_modul < modularity_val:
                best_modul = modularity_val
                best_partition = communities
            i += 1

        modularity_val = best_modul
        best_partition = change_to_key_value(best_partition)

    elif algorithm_name == 'clique_percolation':
        best_partition, modularity_val = clique_percolation(network)
        best_partition = change_to_key_value(best_partition)

    elif algorithm_name == 'louvain':
        best_partition = community_louvain.best_partition(network)
        modularity_val = community_louvain.modularity(best_partition, network)

    communities = set(best_partition.values())
    num_of_partitions = len(communities)
    for node_community in communities:
        partition.append(node_community)
    for node_key, node_community in best_partition.items():
        if isinstance(partition[node_community], int):
            partition[node_community] =[]
        partition[node_community].append(node_key)

    result["num_partitions"] = num_of_partitions
    result["partition"] = partition
    result["modularity"] = modularity_val

    return result


class b_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def community_detection(path):
    """
    for a given path of .gml print partition of the graph according to community_detector function
    for network we calculate the best partition based of the modularity value,
    and saved the best partition in .gml and .gexf files.
    :param path: path to .gml file
    :return: the best partition of network
    """
    print("--" * 80)
    print(f"{b_colors.HEADER} {path} {b_colors.ENDC}")
    print("--" * 80)
    G = nx.read_gml(path)
    res_c = community_detector("clique_percolation", G)
    res_l = community_detector('louvain', G)
    res_gne = community_detector('girvin_newman', G, edge_selector_optimizer)
    res_gn = community_detector('girvin_newman', G)
    max_modularity = max(res_c['modularity'],res_l['modularity'], res_gn['modularity'], res_gne['modularity'])
    max_res = None
    print("*" * 30 + ' clique_percolation ' + "*" * 30)
    if res_c['modularity'] == max_modularity:
        max_res = res_c
        print(f"{b_colors.OKGREEN} {res_c} {b_colors.ENDC}")
    else:
        print(res_l)
    print("*" * 30 + ' louvain ' + "*" * 30)
    if res_l['modularity'] == max_modularity:
        max_res = res_l
        print(f"{b_colors.OKGREEN} {res_l} {b_colors.ENDC}")
    else:
        print(res_l)
    print("*" * 15 + ' girvin_newman -- edge_selector_optimizer ' + "*" * 15)
    if res_gne['modularity'] == max_modularity:
        print(f"{b_colors.OKCYAN} {res_gne} {b_colors.ENDC}")
        max_res = res_gne
    else:
        print(res_gne)
    print("*" * 30 + ' girvin_newman ' + "*" * 30)
    if res_gn['modularity'] == max_modularity:
        max_res = res_gn
        print(f"{b_colors.OKGREEN} {res_gn} {b_colors.ENDC}")
    else:
        print(res_gn)

    full_network = add_community_attribute(max_res["num_partitions"], max_res["partition"], G)
    return full_network


def harry_potter_books_communities(partition=False):
    """
    this function pass the path of book graph to community_detection function
    for calculate the best partition of the given network
    the function saves the community_results_full_book in .gml and .gexf files
    :param partition: true if we want to get community_results_full_book on a divided book שnd false for a full book.
    :return:
    """
    for i in range(1, 8):
        path = f"fullbooks_graphs/Full_Network_Book_{i}"
        if partition:
            for part in range(1, 4):
                path = f"divided_books_graphs/Full_Network_Book_{i}"
                path = f"{path}_part_{part}.gml"
                full_network = community_detection(path)
                nx.write_gml(full_network, f"community_results_partition/partition_Book_{i}_part_{part}.gml")
                nx.write_gexf(full_network, f"community_results_partition/partition_Book_{i}_part_{part}.gexf")
        else:
            path = f"{path}.gml"
            full_network = community_detection(path)
            path
            nx.write_gml(full_network, f"community_results_full_book/partition_Book_{i}.gml")
            nx.write_gexf(full_network, f"community_results_full_book/partition_Book_{i}.gexf")


def add_community_attribute(num_partitions, partition, full_network):
    """
    this function add for each node in network an attribute named 'part'
    the value is a number of the community the node belong to .
    :param num_partitions: int number - the community amount in network
    :param partition: list of lists - list of partition - each value is a list of nodes name
    :param full_network: networkX graph
    :return: networkX graph with 'part' attribute
    """
    for i in range(1, num_partitions+1):
        partition_i = partition[i-1]
        for node_name in partition_i:
            full_network.nodes[node_name]["partition"] = i
    return full_network


if __name__ == "__main__":
    harry_potter_books_communities(True)
    harry_potter_books_communities(False)
