# NLP-Harry-Poter-Books

A NLP project about the connections between the characters in the Harry Potter books
-------------------------------------

'text' directory includes the text books.

'characters_from_books' directory inclueds .pkl files with list of characters accurding to BERT NER characters output.

'related_characters_books' inclueds .pkl files with sentiments between each two characters.

'fullbooks_graphs' directory inclueds .gml files with graph of a full book accurding to ngram output.

'divided_books_graphs' directory inclueds .gml files with graph of a partial books accurding to ngram output.

'community_results_full_book' directory inclueds .gml and .gexf files of graph with an attribute of community for each node in a full book.

'community_results_partition' directory inclueds .gml and .gexf files of graph with an attribute of community for each node in a partial book.

-----------------------------------

'TextProcessing_BERT.py' responsible of text parsing and extraction of characters and the sentiments between characters.

'networks.py' responsible of graph creation accurding to characters senmantic list.

'harry_books_communities.py' responsible of finding communities in the network for each full book and each partial book.

-----------------------------------

