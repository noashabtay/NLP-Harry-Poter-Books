import nltk
import re
import pickle
import collections
from afinn import Afinn
from nameparser import HumanName
from nltk.sentiment import SentimentIntensityAnalyzer
import sys

sys.path.append('/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project')
from BERT_NER.bert import Ner


nltk.download(["vader_lexicon"])


# def convert_characters_dictionary(characters_dict: dict):
#     converted_dict = {}
#     for character in characters_dict:
#         names_list = characters_dict[character]
#         for name in names_list:
#             converted_dict[name] = character
#
#     return converted_dict


def convert_names_of_characters_in_text(characters_dict: dict, text: str):
    """
    Given a dictionary that maps characters' different name forms and their representing forms, swaps every name form
    of a character with its representing form.

    :param characters_dict: a dictionary that maps characters' different name forms and their representing forms.
    :param text: the text of the book.
    :return: the text after the process.
    """
    converted_txt = text
    characters_unique = sorted(set(characters_dict.values()))

    # making new sorted dictionary without equal key and value pairs
    characters_dict_sorted = collections.OrderedDict(sorted(characters_dict.items()))
    characters_dict_sorted_filtered = dict()
    for (k, v) in characters_dict_sorted.items():
        if k != v:
            characters_dict_sorted_filtered[k] = v

    reversed_names = []

    # inserting the converted strings in reverse order in order to avoid
    # duplicate conversion for names with more then 1 word (which can cause harry-> harry potter-> harry harry potter)
    for character in characters_unique:
        reverse = character[::-1]
        converted_txt = converted_txt.replace(character, reverse)
        reversed_names.append(reverse)

    for character in characters_dict_sorted_filtered:
        reverse = characters_dict[character][::-1]
        converted_txt = converted_txt.replace(character, reverse)
        reversed_names.append(reverse)

    # re-reversing the names to their normal form
    for rev in set(reversed_names):
        reverse = rev[::-1]
        converted_txt = converted_txt.replace(rev, reverse)

    return converted_txt


def save_in_file(filename: str, obj):
    """
    Saves the given object in a pickle file.
    """
    with open(filename, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)


def get_entities_bert(book_text: list, book_vocab: set, model):
    personalities = []
    text = ""
    counter = 0
    num_of_batches = 0
    for word in book_text:
        if word == '':
            continue

        text = f"{text} {word}"
        counter += 1
        if counter == 300 or 300 * num_of_batches + counter == len(book_text):
            num_of_batches += 1
            counter = 0

            output = model.predict(text)
            text = ""
            for i in range(len(output)):
                word_tag = output[i]
                if word_tag['tag'] == 'B-PER':
                    personalities.append(word_tag['word'])
                elif word_tag['tag'] == 'I-PER' and output[i - 1]['tag'] == 'B-PER' and len(personalities) > 0 and \
                        word_tag['word'] != personalities[len(personalities) - 1]:
                    personalities[
                        len(personalities) - 1] = f"{personalities[len(personalities) - 1]} {word_tag['word']}"

    characters_counter_dict = {}
    for character in personalities:
        characters_counter_dict[character] = personalities.count(character)

    filtered_characters = [k for k, v in characters_counter_dict.items() if v > 3]

    personalities_set = set(filtered_characters)
    fixed_personalities = []
    add = True
    for per in personalities_set:
        per_words = per.split()
        for per_word in per_words:
            per_word = per_word.lower()
            if per_word in book_vocab:
                add = False
                break
        if add:
            fixed_personalities.append(per)

        add = True

    filtered_characters_dict = {}
    for character in fixed_personalities:
        filtered_characters_dict[character] = characters_counter_dict[character]

    return filtered_characters_dict


# def convert_tuple_to_text(ngram_tuple: tuple):
#     ngram = ""
#     for word in ngram_tuple:
#         ngram = f"{ngram} {word}"
#
#     return ngram


# def get_ngrams_afinn(txt, characters_list, n=10, threshold_names=5, threshold_sentiments=1):
#     characters_dict = {}
#     # characters_list_lower_case = [x.lower() for x in characters_list]
#     dic_index_to_sentiment = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
#     related_characters = []
#     # txt = normalize_text(txt).split()
#     ngrams = nltk.FreqDist(nltk.ngrams(txt, n))
#     af = Afinn()
#     for i in range(0, len(characters_list), 1):
#         for j in range(i + 1, len(characters_list), 1):
#             counter = 0
#             positive, negative = 0, 0
#             for ngram_tuple in ngrams:
#                 ngram = convert_tuple_to_text(ngram_tuple)
#                 if characters_list[i] in ngram and characters_list[j] in ngram:
#                     counter += 1
#                     temp_score = af.score(ngram)
#                     # temp_score = af.score(' '.join(ngram))
#                     # print("temp score :", temp_score)
#                     if temp_score > 0:
#                         positive += temp_score
#                     elif temp_score < 0:
#                         negative += temp_score
#
#             # if positive > negative:
#             #   sentiment = 'Positive'
#             # else:
#             #   sentiment = 'Negative'
#             if negative == 0 or positive == 0:
#                 if negative > 0:
#                     sentiment = 'Negative'
#                 elif positive > 0:
#                     sentiment = 'Positive'
#                 else:
#                     sentiment = 'Neutral'
#
#             else:
#                 if positive / negative > threshold_sentiments:
#                     sentiment = 'Positive'
#                 elif negative / positive > threshold_sentiments:
#                     sentiment = 'Negative'
#                 else:
#                     sentiment = 'Neutral'
#
#             characters_dict[(characters_list[i], characters_list[j])] = (sentiment, counter)
#
#     # print("keys: " ,characters_dict.keys())
#     for pair in characters_dict.keys():
#         # print(characters_dict[pair][1])
#         if characters_dict[pair][1] >= threshold_names:
#             related_characters.append((pair, characters_dict[pair][0], characters_dict[pair][1]))
#
#     return related_characters


def polarity_score(sentence: str):
    """
    Returns the probability of each sentiment to determine a given sentence.
    :return: a tuple with the score of each sentiment.
    """
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(sentence)
    return score['pos'], score['neg'], score['neu']


def get_ngrams(txt, characters_list, n=10, threshold_names=5, threshold_sentiments=1.25):
    characters_dict = {}
    related_characters = []
    ngrams = nltk.FreqDist(nltk.ngrams(txt, n))
    for i in range(0, len(characters_list), 1):
        for j in range(i + 1, len(characters_list), 1):
            counter = 0
            positive, negative = 0, 0
            for ngram in ngrams:
                text_ngram = ' '.join(ngram)
                if characters_list[i] in text_ngram and characters_list[j] in text_ngram:
                    counter += 1
                    temp_score = polarity_score(text_ngram)
                    # positive += temp_score[0]
                    # negative += temp_score[1]
                    if temp_score[0] > temp_score[1]:
                        positive += 1
                    elif temp_score[1] > temp_score[0]:
                        negative += 1

            if negative == 0 or positive == 0:
                if negative > 0:
                    sentiment = 'Negative'
                elif positive > 0:
                    sentiment = 'Positive'
                else:
                    sentiment = 'Neutral'

            else:
                if positive / negative > threshold_sentiments:
                    sentiment = 'Positive'
                elif negative / positive > threshold_sentiments:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'

            characters_dict[(characters_list[i], characters_list[j])] = (sentiment, counter)

    for pair in characters_dict.keys():
        if characters_dict[pair][1] >= threshold_names:
            related_characters.append((pair, characters_dict[pair][0], characters_dict[pair][1]))

    return related_characters


def get_the_same_character_name(intersection, popularity):
    """
    This function is based on the output of get_intersection_names function.
    The function gets as an input the intersection of names and dict of characters with their popularity.
    return the unique names based on the given intersection.

    :param intersection: list of intersection names (the output of get_intersection_names function)
                        ['George Weasley', 'Weasley', 'George', 'Ron Weasley', 'Ron']
    :param popularity: dict of names with popularity  - {"George Weasley": 100, "Harry": 8606 , "Ginny": 90, ...}
    :return: {
                'George Weasley': ['George Weasley', 'Weasley', 'George'],
                'Ron Weasley': ['Ron Weasley', 'Weasley', 'Ron']
            }
    """
    unique_names = {}
    for name in intersection:
        if len(name.split(" ")) > 1:
            name_params = HumanName(name)
            unique_name = get_max_of_name(name, name_params, popularity)
            if unique_name in unique_names:
                if name not in unique_names[unique_name]:
                    unique_names[unique_name].append(name)
            else:
                unique_names[unique_name] = [name]
    for name in intersection:
        found = False
        for unique_name in unique_names:
            if name in unique_names[unique_name]:
                found = True
                break
        if not found:
            name_params = HumanName(name)
            new_unique_name = get_max_of_name(name, name_params, popularity)
            for unique_name in unique_names:
                if new_unique_name in unique_name.split(" "):
                    unique_names[unique_name].append(new_unique_name)

    for name in intersection:
        found = False
        for unique_name in unique_names:
            if name in unique_names[unique_name]:
                found = True
                break
        if not found:
            name_params = HumanName(name)
            new_unique_name = get_max_of_name(name, name_params, popularity)
            unique_names[new_unique_name] = [name]

    return unique_names


def get_max_of_name(name, name_params, poplarity):
    """
    this part return unique name based on the popularity -
    for example if the popularity rank of Ron Weasley is 100 and the popularity rank of Ginny Weasley is 300
    so the final unique name of Weasley ond Mr. Weasley will be Ginny Weasley
    :param name: name of character
    :param name_params: name params of the character -
    is the output of nameparser.HumanName and contains: last = last name, first = first name.
    :param poplarity: dict of names with popularity  - {"George Weasley": 100, "Harry": 8606 , "Ginny": 90, ...}
    :return:

    """
    first_name = name_params.first
    last_name = name_params.last
    if first_name != "" and last_name != "":
        full_name = f"{first_name} {last_name}"
    elif first_name != "":
        full_name = first_name
    else:
        full_name = last_name
    if full_name not in poplarity and last_name not in poplarity and first_name not in poplarity:
        full_name = name
    return full_name


def get_intersection_names(names, name):
    """
    This function returns for a given name a list of other names that contains given name
    or names in which the given name contains them.
    for example
    for the list of names: ["Mr. Weasley", "potter", "Ron Weasley", "harry", "Ginny", Harry potter", "Ginny Weasley"]
    and for name  = "harry"
    the result should be - ["potter", "harry", Harry potter"]
    and for ane = "Weasley"
    the result should be - ["Mr. Weasley", "Ginny", "Ginny Weasley", "Ron Weasley"]

    :param names: list ["Mr. Weasley", "[otter", "Ron Weasley", "harry", "Ginny", Harry potter", "Ginny Weasley", ...]
    :param name: str "Potter"
    :return: a list of name intersection..
    """
    intersection = []
    tmp = {}
    not_intersection = {}
    found_intersection = False

    for name_2 in names:
        for word in name.split(" "):
            tmp[word] = ""
            if (name_2 == word) or (name_2.find(word) >= 0):  # name_2 = Potter, name = Harry Potter
                found_intersection = True
                if name_2 not in intersection:
                    intersection.append(name_2)
            for n2 in name_2.split(" "):
                if word == n2 or name.find(n2) >= 0:  # name_2 = Mr. Harry Potter, name = Hary Potter
                    found_intersection = True
                    if name_2 not in intersection:
                        intersection.append(name_2)
            for n2 in name_2.split(" "):
                for inter_name in intersection:
                    if (n2.find(inter_name) >= 0) or (inter_name.find(n2) >= 0):
                        found_intersection = True
                        if name_2 not in intersection:
                            intersection.append(name_2)
        if not found_intersection:
            not_intersection[name_2] = ""
    for inter in intersection:
        for word in inter.split(" "):
            if word in not_intersection:
                intersection.append(word)
                del not_intersection[word]
    return intersection


def get_unique_names_dictionary(names: list, popularity: dict):
    """
    This function get as input list of characters names from BERT module and the popularity of them in the book section.
    for the given params the function creates a dictionary of unique characters names.
    for example the characters: "Harry", "Mr. Potter", "Harry Potter" - will consider as "Harry Potter"

    :param names: list of names from BERT module - ["George Weasley", "Harry", "Ginny", ...]
    :param popularity: dict of names with popularity  - {"George Weasley": 100, "Harry": 8606 , "Ginny": 90, ...}
    :return: dict-  {"George Weasley": "George Weasley",
                    "Harry": "Harry Potter",
                    "Ginny": "Ginny",
                    "Mr. Potter": "Harry Potter",
                    "Harry Potter" : "Harry Potter" ...}
    """
    res = {}
    for name in names:
        intersection = get_intersection_names(names, name)
        unique_names = get_the_same_character_name(intersection, popularity)

        """
        this part return unique name based on the popularity -
        for example if the popularity rank of Ron Weasley is 100 and the popularity rank of Ginny Weasley is 300 
        so the final unique name of Weasley ond Mr. Weasley will be Ginny Weasley        
        """
        for unique_name, nicknames in unique_names.items():
            for nickname in nicknames:
                if nickname in res:
                    res[nickname] = unique_name if (popularity[unique_name] > popularity[res[nickname]]) else res[
                        nickname]
                else:
                    res[nickname] = unique_name
    """
    this part updates the new popularity after the consideration of different names into one name
    for example if the popularity rank of Ron Weasley is 100 and the popularity rank of Ron is 300 
    so the final unique name of Ron Weasley with popularity 100+300 = 400        
    """
    new_popularity = {}
    for name, unique_name in res.items():
        if unique_name in new_popularity and unique_name != name:
            new_popularity[unique_name] += popularity.get(name, 0)
        elif unique_name not in new_popularity and unique_name != name:
            new_popularity[unique_name] = popularity.get(name, 0) + popularity[unique_name]
        else:
            new_popularity[unique_name] = popularity[unique_name]

    for name, unique_name in res.items():
        name_params = HumanName(name)
        if name != unique_name and name_params.title != "":
            count_max = 0
            name_max = ""
            for entity in new_popularity:
                name_params = HumanName(name)
                splited = entity.split(" ")
                if len(splited) > 1:
                    if name_params.last == splited[1] or name_params.first == splited[0]:
                        count_max = new_popularity[entity] if new_popularity[entity] >= count_max else count_max
                        name_max = entity if new_popularity[entity] >= count_max else name_max
            if name_max != "":
                res[name] = name_max
                res[unique_name] = name_max
                new_popularity[name_max] += new_popularity.get(unique_name, 0)

                if unique_name in new_popularity:
                    del new_popularity[unique_name]
    # print(res)
    # print(len(new_popularity))
    # print(len(res))
    # print(len(set(res.values())))
    # print(res)
    return res


def get_character_relations_from_book(book_path: str, run_bert: bool, ngram_size=10, threshold_names=5, threshold_sentiments=1.25):
    """
    Extracts the characters in the book and analyzes the relations between them.
    BERT is used for entities extraction and nameparser module for uniting the different names of the same character
    into one representing form. Sentiment analysis is used to define the sentiment of the relation between each
    character pair.

    :param book_path: path to the book
    :param run_bert: True in order to run BERT model on the text, False in order to read BERT's output from a pickle
    file. At the first run on a specific book, True is required.
    :return: A list of tuples, each tuple contains the characters pair, the sentiment of their relation and the number
    of their common appearances.
    """

    with open(book_path, 'r') as book_1:
        book_text = book_1.read()

    # book_1_text = book_1.read().split()
    # d = ",.!?/&-:;@—'‘“”...…’ \n"
    d = "!?/&-:;@—'‘“”...…’ \n"
    book_text = re.split("[" + "\\".join(d) + "]", book_text)
    book_text_vocab = set(book_text)

    if run_bert:
        # print("something is wrong !!!!!")
        model = Ner("/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project/BERT_NER/out_base/")
        characters_dict = get_entities_bert(book_text, book_text_vocab, model)
        save_in_file('characters_from_book_1.pkl', characters_dict)

    else:
        with open('characters_from_book_1.pkl', 'rb') as pkl_file:
            characters_dict = pickle.load(pkl_file)

    book_text_str = ' '.join(book_text)
    characters_list = list(characters_dict.keys())
    unique_names_dict = get_unique_names_dictionary(characters_list, characters_dict)
    converted_text = convert_names_of_characters_in_text(unique_names_dict, book_text_str)
    unique_characters_list = list(set(unique_names_dict.values()))

    related_characters = get_ngrams(converted_text.split(), unique_characters_list, ngram_size, threshold_names, threshold_sentiments)
    return related_characters


book_path = '/Users/noashabtay/Desktop/studies/sem 8/nlp/project/parse_dialog-master/Harry Potter and the Sorcerer\'s Stone.txt'
related_characters = get_character_relations_from_book(book_path, False)
save_in_file('related_characters_book_1', related_characters)
