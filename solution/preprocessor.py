import nltk
from nltk.stem.porter import *
import os
import ast

# Stopword list - 30
stop_words_30 = ['the',  'and', 'of', 'in', 'to', 'a', 'for', 'is', 'on', 'with', 'that', 'research',
              'at', 'as', 'by', 'are', 'from', 'this', 'an', 'be', 'will', 'concordia', 'university',
              'j', 'de', 'we', 'students', 'i', 'or', 'science']


def get_files(path):
    """
    :param path: path of folder that need to be scanned
    :return: list of sgm files in the specific folder
    """
    file_list = []
    with os.scandir(path) as folder:
        for file in folder:
            if file.is_file() and file.name.endswith('txt'):
                file_list.append(file.path)
    return file_list


def remove_special_characters(words):
    '''Removes punctuations, linefeed/carriage return and other non-alphanumeric characters'''
    special_characters = '!?"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

    transtable = str.maketrans('', '', special_characters)
    processed_words = [term.translate(transtable) for term in words]  # remove punctuations
    useless_words = ['','s','-','--']
    processed_words = [term for term in processed_words if term not in useless_words]
    return processed_words

def remove_numbers(words):
    '''Remove numbers'''
    processed_words = [term for term in words if not term.isdigit()]
    return processed_words

def case_folding(words):
    '''Case Folding'''
    processed_words = [term.lower() for term in words]
    return processed_words

def remove_stopwords(stopwords, words):
    '''Remove stopwords'''
    processed_words = [term for term in words if not term in stopwords]
    return processed_words

def stemming(terms):
    '''Stemming'''
    stemmer = PorterStemmer()
    stemmed_terms = [stemmer.stem(term) for term in terms]
    return stemmed_terms

def tokenize(file_list):
    """"
    :param stop_words_150: flag to remove 150 stopwords
    :param stop_words_30: flag to remove 30 stopwords
    :param case_folding: flag to change to lowercase
    :param no_numbers: flag to remove numbers
    :param file_list: list of file that need to be tokenize
    :return: list of all tokens
    """
    token_id_pairs = []
    for file_index, file in enumerate(file_list):
        print('Processing ' + file_list[file_index] + '...')
        with open(file, 'r') as file_obj:
            data = file_obj.read()
        if data.__len__() == 0:
            continue
        info = ast.literal_eval(data)
        newid = info['title']
        body = info['content']
        tokens = nltk.word_tokenize(body)
        # Preprocess tokens, aka lazy compression
        tokens = remove_special_characters(tokens)
        tokens = remove_numbers(tokens)
        # tokens = case_folding(tokens)
        # tokens = remove_stopwords(stop_words_30, tokens)
        # tokens = stemming(tokens)
        for token in tokens:
            token_id_pairs.append((token, newid))
        print('Tokenization for ' + file_list[file_index] + ' completed...')
    print('Tokenization for all files completed...')
    return token_id_pairs

