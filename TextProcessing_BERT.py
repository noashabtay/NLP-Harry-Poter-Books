import nltk
import re
import pickle
import collections
from nameparser import HumanName
from nltk.sentiment import SentimentIntensityAnalyzer
import sys
import string
import math
import statistics

# sys.path.append('/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project')
# from BERT_NER.bert import Ner # need to import bert project

nltk.download(["vader_lexicon"])


def book_division(book_path, num_of_division):
    """
    This function divide the given book path to a given num of division.
    :param book_path: str path to book
    :param num_of_division: int
    :return: list od str
    """
    with open(book_path, 'rt') as book_text:
        text = ''
        divisions = []
        for i in range(0, num_of_division):
            divisions.append('')
        for line in book_text:
            text = text + line
        splited = text.split("Chapter")
        num_of_chapters_in_one_division = len(splited)/num_of_division
        num_of_chapters_in_one_division = int(math.ceil(num_of_chapters_in_one_division))

        for division_index in range(0, num_of_division):
            from_index = num_of_chapters_in_one_division * division_index
            to_index = num_of_chapters_in_one_division
            if division_index != 0:
                from_index = num_of_chapters_in_one_division*division_index
                to_index = num_of_chapters_in_one_division*(division_index+1)
            if to_index > len(splited):
                to_index = len(splited)
            for i in range(from_index, to_index):
                divisions[division_index] += "Chapter" + splited[i]
        return divisions


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
    """
    Gets a list of the tokens in the book, the vocabulary and the BERT model, and generates a dictionary
    with the different entities detected by the BERT model and the amount of times each one appeared in the book.
    """
    personalities = []
    text = ""
    counter = 0
    num_of_batches = 0
    for word in book_text:
        if word == '':
            continue

        # creates a 250 words long batches as an efficient input for the model
        text = f"{text} {word}"
        counter += 1
        if counter == 250 or 250 * num_of_batches + counter == len(book_text):
            num_of_batches += 1
            counter = 0

            output = model.predict(text)
            text = ""
            for i in range(len(output)):
                # removing punctuations - this is done in this step for better entity detection
                word_tag = output[i]
                word = word_tag['word'].translate(str.maketrans(dict.fromkeys(string.punctuation)))
                if word in "!?/&-:;@—'‘“”…’ \n":
                    continue

                # composition of the entities and insertion into a list
                if word_tag['tag'] == 'B-PER':
                    personalities.append(word)
                elif word_tag['tag'] == 'I-PER' and output[i - 1]['tag'] == 'B-PER' and len(personalities) > 0 and \
                        word != personalities[len(personalities) - 1]:
                    personalities[
                        len(personalities) - 1] = f"{personalities[len(personalities) - 1]} {word}"

    # counting entities appearances
    characters_counter_dict = {}
    for character in personalities:
        characters_counter_dict[character] = personalities.count(character)

    # removing insignificant characters
    filtered_characters = [k for k, v in characters_counter_dict.items() if v > 3]

    # removing false positives, i.e. words tagged as entities and shouldn't have
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


def polarity_score(sentence: str):
    """
    Returns the probability of each sentiment to determine a given sentence.
    :return: a tuple with the score of each sentiment.
    """
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(sentence)
    return score['pos'], score['neg'], score['neu']


def get_ngrams(txt, characters_list, n=10, threshold_names=5, threshold_sentiments=1.25):
    """
    Gets the text of the book after name occurences conversion, a list of characters in the text, the N-gram size,
    a threshold for common appearances to be considered as a relation, and a threshold for the sentiment determination,
    and finds the relations between the characters. The algorithm goes over the N-grams and looks for character pairs in
    it, count the amount of N-grams each pair has and classifies each one's sentiment. In case the proportion between
    the sentiments exceeds the sentiment threshold, the relation is classified as the larger sentiment, else neutral.
    """
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
                    # classifies the ngram's sentiment
                    temp_score = polarity_score(text_ngram)
                    if temp_score[0] > temp_score[1]:
                        positive += 1
                    elif temp_score[1] > temp_score[0]:
                        negative += 1

            # classifies the sentiment of the whole relation
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
        # reduces insignificant relations from the final list
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


def get_character_relations_from_book(book_path: str, run_bert: bool, num_divisions:int = None, ngram_size=10, threshold_names=25, threshold_sentiments=1.25):
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
    # d = "!?/&-:;@—'‘“”…’ \n"
    d = "’ "
    book_text = re.split("[" + "\\".join(d) + "]", book_text)
    # book_text = book_text.split()
    book_text_vocab = set(book_text)

    if run_bert:
        model = Ner("/Users/royjudes/Documents/Studies/Sem B/Social Dynamics/Project/BERT_NER/out_base/")
        characters_dict = get_entities_bert(book_text, book_text_vocab, model)
        save_in_file('characters_from_book_7.pkl', characters_dict)

    else:
        with open('characters_from_book_7.pkl', 'rb') as pkl_file:
            characters_dict = pickle.load(pkl_file)

    characters_list = list(characters_dict.keys())
    unique_names_dict = get_unique_names_dictionary(characters_list, characters_dict)
    unique_characters_list = list(set(unique_names_dict.values()))

    if num_divisions is not None:
        related_characters = []
        book_divisions = book_division(book_path, num_divisions)
        for i in range(num_divisions):
            converted_text_book_part = convert_names_of_characters_in_text(unique_names_dict, book_divisions[i])
            related_characters_from_division = get_ngrams(converted_text_book_part.split(), unique_characters_list, ngram_size, 1, threshold_sentiments)
            appearances = []
            for pair in related_characters_from_division:
                appearances.append(pair[2])

            mean_common_appearances = statistics.mean(appearances)
            std_common_appearances = statistics.stdev(appearances)
            normalized_appearances = [((appearances[j] - mean_common_appearances) / std_common_appearances) for j in range(len(appearances))]
            min_normalized_appearances = min(normalized_appearances)
            max_normalized_appearances = max(normalized_appearances)

            filtered_related_characters = []
            for j in range(len(normalized_appearances)):
                if normalized_appearances[j] - min_normalized_appearances >= max_normalized_appearances * 0.05:
                    filtered_related_characters.append(related_characters_from_division[j])

            related_characters.append(filtered_related_characters)

        return related_characters

    else:
        book_text_str = ' '.join(book_text)
        converted_text = convert_names_of_characters_in_text(unique_names_dict, book_text_str)


    related_characters = get_ngrams(converted_text.split(), unique_characters_list, ngram_size, threshold_names, threshold_sentiments)
    return related_characters


def statistic_characters():
    """
    this function prints statistic we wanted to show in the report -
    how many characters in each book
    how many nicknames names are for each characters
    :return:
    """
    for i in range(1, 8):
        path = f"characters_from_books/characters_from_book_{i}.pkl"
        with open(path, 'rb') as characters_file:
            characters_dict = pickle.load(characters_file)
            characters_list = list(characters_dict.keys())
            unique_names_dict = get_unique_names_dictionary(characters_list, characters_dict)
            unique_characters_list = list(set(unique_names_dict.values()))
            print(f"book number: {i} , unique characters: {len(unique_characters_list)}")
            for unique_name in unique_characters_list:
                print(f"{unique_name=}")
                for name in unique_names_dict:
                    if unique_names_dict[name] == unique_name:
                        print(name)
                print("-"*50)
