import os
import ast
import nltk
import re
import math
from preprocessor import get_files

# N represents the total number of documents
N = len([f for f in os.listdir('extracted_files') if os.path.isfile(os.path.join('extracted_files', f))])
final_index_dictionary = {}


def convert_index_dictionary():
    with open('FINAL_INDEX.txt', 'r') as final_index_file:
        for line in final_index_file:
            new_line = (re.sub('\'|:|\[|,|\]', ' ', line)).split()
            final_index_dictionary[new_line[0]] = new_line[1:]


def sentiment_aggr_function(some_text):
    """
    :param some_text: some text in which the individual words containing a sentiment value will be summed
    :return: the aggregated sentiment score for the text
    """
    afinn = dict(map(lambda p: (p[0], int(p[1])), [line.split('\t') for line in open("AFINN-111.txt")]))
    sentiment_aggr = sum(map(lambda word: afinn.get(word, 0), some_text.lower().split()))
    return sentiment_aggr


def get_document_tf_idf(term, document):
    """
    :param term: a term
    :param document: the title of the document
    :return: the tf-idf of the term in document
    """
    if term in final_index_dictionary.keys():
        if document in final_index_dictionary[term]:
            index = final_index_dictionary[term].index(document)
            return float(final_index_dictionary[term][index + 3])
        else:
            return 0.0
    return 0.0


def find_q_tf_idf(term, term_frequency):
    """
    :param term: a term in the query
    :param term_frequency: the frequency of the term in the query
    :return: the tf-idf of the term
    """
    if term in final_index_dictionary.keys():
        return (1 + math.log(term_frequency)) * (math.log(N / ((final_index_dictionary[term].__len__()) / 4)))
    return 0.0


def q_d_square_sum_function(query_dictionary, d_body, d_title):
    """
    :param query_dictionary: dictionary containing the terms of a query
    :param d_body: body content of a document
    :param d_title: title of a document
    :return: the denominator of the cosine similarity function
    """
    q_square_sum = 0.0
    d_square_sum = 0.0

    for key in query_dictionary.keys():
        q_square_sum += query_dictionary[key] * query_dictionary[key]

    for d_term in nltk.word_tokenize(d_body):
        d_square_sum += get_document_tf_idf(d_term, d_title)*get_document_tf_idf(d_term, d_title)

    return math.sqrt(q_square_sum)*math.sqrt(d_square_sum)


def q_d_sum_function(query_dictionary, d_title):
    """
    :param query_dictionary: dictionary containing the terms of a query
    :param d_title: title of a document
    :return: the numerator of the cosine similarity function
    """
    q_d_sum = 0.0
    for key in query_dictionary.keys():
        q_d_sum += query_dictionary[key] * get_document_tf_idf(key, d_title)
    return q_d_sum


def cosine_similarity(query_dictionary, d_body, d_title):
    """
    :param query_dictionary: dictionary containing the terms of a query
    :param d_body: body content of a document
    :param d_title: title of a document
    :return: the cosine similarity measure
    """
    return q_d_sum_function(query_dictionary, d_title) / q_d_square_sum_function(query_dictionary, d_body, d_title)


def w_s_partial_function(doc_w, doc_s, query_s):
    """
    :param doc_w: tf-idf based on cosine similarity
    :param doc_s: sentiment value of a document
    :param query_s: sentiment value of a query
    :return: the weight of the document
    """
    return doc_w / (1 + (doc_s - query_s) * (doc_s - query_s))


if __name__ == '__main__':
    convert_index_dictionary()
    document_dictionary = {}
    document_url_dictionary = {}
    for file_index, file in enumerate(get_files('extracted_files')):
        with open(file, 'r') as file_obj:
            data = file_obj.read()
        if data.__len__() == 0:
            continue
        info = ast.literal_eval(data)
        document_id = info['title']
        body = info['content']
        url = info['url']
        document_dictionary[document_id] = body
        document_url_dictionary[document_id] = url

    query = input('Please enter a query: ')
    q_tokens = nltk.word_tokenize(query)
    q_dictionary = {}
    for token in q_tokens:
        q_tf = q_tokens.count(token)
        q_tf_idf = find_q_tf_idf(token, q_tf)
        q_dictionary[token] = q_tf_idf

    document_w_s_weight = {}
    q_s = sentiment_aggr_function(query)
    for document_id in document_dictionary.keys():
        d_w = cosine_similarity(q_dictionary, document_dictionary[document_id], document_id)
        d_s = sentiment_aggr_function(document_dictionary[document_id])
        document_w_s_weight[document_id] = w_s_partial_function(d_w, d_s, q_s)

    ranked_document_file = open('RANKED_DOCUMENT_Q3.txt', 'w+')
    sorted_by_value = sorted(document_w_s_weight.items(), key=lambda kv: kv[1], reverse=True)
    rank_number = 0
    ranked_document_file.write("Query: " + query + "\n\n")
    for elements in sorted_by_value:
        rank_number += 1
        ranked_document_file.write("Rank: " + str(rank_number) + " url: " + document_url_dictionary[elements[0]] + " document title: " + elements[0] + " score: " + str(elements[1]) + "\n")
