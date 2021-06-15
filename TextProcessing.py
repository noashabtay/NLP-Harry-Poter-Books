from gensim.models.word2vec import Word2Vec
import nltk
from nltk import tokenize
import csv
from multiprocessing import cpu_count
import re
from numpy import dot
from numpy.linalg import norm
import random


# def create_language_model_for_book(book_file_path: str, vec_size: int = 300, min_count: int = 0, window: int = 5):
#     text = ""
#     with open(book_file_path, 'r') as book_file:
#         text = book_file.read()
#
#     sentences = tokenize.sent_tokenize(text)
#     normalized_sentences = []
#     for sentence in sentences:
#         normalized_sentence = []
#         sentence = sentence.split()
#         for word in sentence:
#             word = normalize_word(word)
#             if word != "":
#                 normalized_sentence.append(word)
#
#         normalized_sentences.append(normalized_sentence)
#
#     # tokenized_sentences = [normalize_text(sentence) for sentence in sentences]
#     # tokenized_text = tokenize.word_tokenize(text)
#     language_model = Word2Vec(sentences=normalized_sentences, window=window, min_count=min_count, vector_size=vec_size, workers=cpu_count())
#     return language_model
#
#
# def create_edges_list(language_model):
#     similarities = []
#     # language_model.wv.index_to_key
#     characters = ['harry', 'ron', 'hermione', 'dumbledore', 'hagrid', 'voldemort', 'draco', 'snape', 'quirrell', 'ginny', 'neville', 'mcgonagall', 'vernon', 'petunia', 'broom']
#     for i in range(0, len(characters), 1):
#         for j in range(i+1, len(characters), 1):
#             # if character == character2:
#             #     continue
#
#             character = characters[i]
#             character2 = characters[j]
#             vec_1 = language_model.wv[character]
#             vec_2 = language_model.wv[character2]
#             similarity = dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))
#             print(character, character2, similarity)
            # similarities.append(similarity)
            # if len(similarities) == 100000:
            #     return similarities

    # return similarities


def normalize_text(text):
    """
    Returns a normalized version of the specified string.
    You can add default parameters as you like (they should have default values!)
    You should explain your decisions in the header of the function.

    Args:
        text (str): the text to normalize

    Returns:
        string. the normalized text.
    """
    words_list = re.findall(r'(?x)(?:[A-Z]\.)+|\w+(?:-\w+)*|\$?\d+(?:\.\d+)?%?', text.lower())  # a regex for extracting tokens from the text
    normalized_text = ""
    for word in words_list:
        normalized_text += f"{word} "

    return normalized_text


def normalize_word(word):
    try:
        return re.findall(r'(?x)(?:[A-Z]\.)+|\w+(?:-\w+)*|\$?\d+(?:\.\d+)?%?', word.lower())[0]

    except:
        return ""


# path = '/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project/text/Harry Potter and the Sorcerer\'s Stone.txt'
# language_model = create_language_model_for_book(book_file_path=path, window=3, vec_size=100)
# create_edges_list(language_model)
# list_a = create_edges_list(language_model)
# random.shuffle(list_a)
# # list_b = sorted(list_a)
# for i in range(100):
#     print(list_a[i])

# counter = 0
# with open(path, 'r') as book_file:
#     text = book_file.read().split()
#     for word in text:
#         word = normalize_word(word)
#         if word == 'alley':
#             counter += 1
#
#     print(counter)


# --------------------------------------------------- NGRAMS -----------------------------------------------------------

def get_related_characters(book_file_path: str, characters_list: list, n: int = 10, threshold: int = 50):
    with open(book_file_path, 'r') as book_file:
        text = book_file.read()

    characters_dict = {}
    related_characters = []
    text = normalize_text(text).split()
    ngrams = nltk.FreqDist(nltk.ngrams(text, n))
    for i in range(0, len(characters_list), 1):
        for j in range(i+1, len(characters_list), 1):
            counter = 0
            for ngram in ngrams:
                if characters_list[i] in ngram and characters_list[j] in ngram:
                    counter += 1

            characters_dict[(characters_list[i], characters_list[j])] = counter

    for pair in characters_dict.keys():
        if characters_dict[pair] >= threshold:
            related_characters.append(pair)

    return related_characters


def load_characters_into_list(characters_file_path: str):
    characters_list = []
    with open(characters_file_path, 'r') as characters_file:
        characters = characters_file.read().split()
        for character in characters:
            characters_list.append(character)

    return characters_list


path = '/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project/text/Harry Potter and the Sorcerer\'s Stone.txt'
characters_file_path = '/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project/characters.txt'
characters_list = load_characters_into_list(characters_file_path)
related_characters = get_related_characters(path, characters_list)

# counter = 0
# with open(path, 'r') as book_file:
#     text = book_file.read()
#     text = normalize_text(text)
#     text = text.split()
#     ngrams = nltk.FreqDist(nltk.ngrams(text, 10))
#     for ngram in ngrams:
#         if 'ron' in ngram and 'hermione' in ngram:
#             counter += 1
#
#     print(counter)

