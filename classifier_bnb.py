from random import random
import math

UNKNOWN = "{--UnK--}"


def count_labels(labels):
    return {label: sum(1 for l in labels if l == label) for label in set(labels)}


def preprocessed(text_to_process):
    lower_case_text = text_to_process.lower()
    processed_text = ""
    for symbol in lower_case_text:
        if "a" <= symbol <= "z" or "а" <= symbol <= "я" or "0" <= symbol <= "9" or symbol == " " or symbol == "\n":
            processed_text += symbol
        else:
            processed_text += " " + symbol + " "
    return processed_text.strip().replace("  ", " ").replace("\t", "")


def normalize(text_to_normalize):
    words_to_add = text_to_normalize.split(" ")
    bag_of_words = set(words_to_add)
    for ind, word in enumerate(words_to_add):
        if ind != 0:
            bag_of_words.add(words_to_add[ind - 1] + " " + words_to_add[ind])
    return bag_of_words


def train(train_texts, train_labels, pretrain_params=None):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    label_types = sorted(list(set(train_labels)))
    analyzed_texts = {}

    for label in train_labels:
        if label not in analyzed_texts:
            analyzed_texts[label] = []

    for ind, text in enumerate(train_texts):
        corresponding_label = train_labels[ind]
        needed_texts = analyzed_texts[corresponding_label]
        needed_texts.append(preprocessed(text))

    labeled_word_frequency = {}
    total_word_occurrences = {}
    for label in train_labels:
        if label not in labeled_word_frequency:
            labeled_word_frequency[label] = {}

    for label in analyzed_texts.keys():
        for text_ind, text in enumerate(analyzed_texts[label]):
            analyzed_texts[label][text_ind] = normalize(text)

            words_list = analyzed_texts[label][text_ind]
            for word in words_list:
                if word not in labeled_word_frequency[label]:
                    labeled_word_frequency[label][word] = 1
                else:
                    labeled_word_frequency[label][word] += 1

                if word not in total_word_occurrences:
                    total_word_occurrences[word] = 1
                else:
                    total_word_occurrences[word] += 1

    total_word_occurrences[UNKNOWN] = 1
    sorted_for_cut = list(sorted(total_word_occurrences.values()))
    cut_point = sorted_for_cut[len(sorted_for_cut) // 20]
    for label in labeled_word_frequency.keys():
        labeled_word_frequency[label][UNKNOWN] = 1
        labeled_word_frequency_keys = labeled_word_frequency[label].copy().keys()
        for word in labeled_word_frequency_keys:
            if total_word_occurrences[word] <= cut_point:
                labeled_word_frequency[label][UNKNOWN] += labeled_word_frequency[label][word]
                total_word_occurrences[UNKNOWN] += labeled_word_frequency[label][word]

                del labeled_word_frequency[label][word]

    for label in labeled_word_frequency.keys():
        for word in labeled_word_frequency[label]:
            labeled_word_frequency[label][word] /= total_word_occurrences[word]

    num_label0 = 0
    num_label1 = 0
    for label in train_labels:
        if label == label_types[0]:
            num_label0 += 1
        if label == label_types[1]:
            num_label1 += 1

    return {"only_unary": labeled_word_frequency, "relation": num_label0 / num_label1, "labels": label_types}


def pretrain(texts):
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts: a list of texts (str objects), one str per example
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """

    return None


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    processed_texts = []
    for text in texts:
        processed_texts.append(preprocessed(text))

    bags_of_words = []
    for text in processed_texts:
        bags_of_words.append(normalize(text))

    relation = params["relation"]
    labeled_word_frequency = params["only_unary"]
    label_types = params["labels"]

    labeled_word_set = {}
    for label in labeled_word_frequency.keys():
        labeled_word_set[label] = set(labeled_word_frequency[label].keys())

    def prob(arg_word, label_type):
        if arg_word not in labeled_word_set[label_type]:
            return labeled_word_frequency[label_type][UNKNOWN]
        else:
            return labeled_word_frequency[label_type][arg_word]

    predicted_labels = []
    for bag in bags_of_words:
        log_of_plausibility = math.log(relation)
        for word in bag:
            log_of_plausibility += math.log(prob(word, label_types[0]) / prob(word, label_types[1]) + 10**(-7))

        if log_of_plausibility > 0:
            predicted_labels.append(label_types[0])
        else:
            predicted_labels.append(label_types[1])

    return predicted_labels
