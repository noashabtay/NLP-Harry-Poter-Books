import TextProcessing
import SocialNetworks


path = '/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project/text/Harry Potter and the Sorcerer\'s Stone.txt'
characters_file_path = '/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project/characters.txt'
characters_list = TextProcessing.load_characters_into_list(characters_file_path)
related_characters = TextProcessing.get_related_characters(path, characters_list)
graph = SocialNetworks.construct_network_from_neighbours_list(related_characters)
SocialNetworks.draw_graph(graph)
