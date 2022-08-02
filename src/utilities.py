import spacy

import nltk
from nltk.corpus import stopwords
# Need to uncomment the following line in order to download nltk stopwords:
# nltk.download('stopwords')
from textacy.extract import keyword_in_context

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import pandas as pd
import numpy as np
import random

import regex as re
import glob

from collections import Counter

def import_dataset(encoding=None):
    """Imports dataset in csv format"""
    file = glob.glob('data/data_set.csv')
    file = ' '.join(file)
    with open(file, encoding='utf-8') as inputfile:
        col_list = ['text', 'class']
        df = pd.read_csv('data/data_set.csv', usecols=col_list)
    return df


################### Text-Processing Steps ###################

# Load greek language model from spacy:
nlp = spacy.load("el_core_news_md")

# Variables used to remove greek stopwords:
stopwords_lower = list(map(lambda x: x.lower(), stopwords.words('greek')))
STOPWORDS_GREEK = set(stopwords_lower)

# Variable used to remove final 'ς' character:
CHAR_TO_REMOVE = 'ς'


def remove_punctuation(text):
    """Custom function to remove the punctuation. Returns a string."""
    PUNCTUATION_TO_REMOVE = '–«!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~»●·’“”'
    return text.translate(str.maketrans('', '', PUNCTUATION_TO_REMOVE))


def tokenize_regex(text):
    """The following expression matches tokens consisting of at least one letter (\p{L}), 
    preceded and followed by an arbitrary sequence of alphanumeric characters 
    (\w includes digits, letters, and underscore) and hyphens (-). Returns a string."""
    return re.findall(r'[\w-]*\p{L}[\w-]*', text)


def tokenize(text):
    """Use spacy's tokenizer. Returns a string."""
    doc = nlp.tokenizer(' '.join(text))
    return ' .'.join([token.text for token in doc])


def drop_single_letter_words(text):
    return ' '.join([w for w in text if len(w) > 1])


def drop_numbers(text):
    """Custom function to remove numbers. Returns a string."""
    text_wo_numbers = re.sub(r'[0-9]+', '', text)
    return text_wo_numbers


def lemmatize(text):
    """Custom function to lemmatize text. Returns a list with lemmas."""
    doc = nlp(' '.join(text))
    return [token.lemma_ for token in doc]


def process(text, pipeline):
    """Custom function to process a pipeline on text."""
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens


def import_additional_greek_stopwords(STOPWORDS_GREEK):
    """Helper function used add more stopwords to NLTK stopwords list."""
    additional_stopwords = open('data/additional_stopwords.txt', 'r')
    for line in additional_stopwords:
        words = line.strip()
        STOPWORDS_GREEK.add(words)
    return STOPWORDS_GREEK


def remove_stopwords(text):
    """Custom function to remove the stopwords. Returns a string"""
    return ' '.join([word for word in str(text).split() if not word in STOPWORDS_GREEK])

def remove_intonation(text):
    """Custom function to remove word-intonation"""
    rep = {"ά": "α", "έ": "ε", "ή": "η", "ί": "ι", "ό": "ο", "ύ": "υ", "ώ": "ω", "ϊ": "ι",
           "ἀ": "α", "ἐ": "ε", "ἤ": "η", "ἰ": "ι", "ἄ": "α", "ὐ": "υ", "ὡ": "ω", "ὦ": "ω",
           'ὖ': 'υ', 'ὅ': 'ο', 'ῆ': 'η', 'ῇ': 'η', 'ῦ': 'υ', 'ὁ': 'ο', 'ὑ': 'υ', 'ὲ': 'ε',
           'ὺ': 'υ', 'ἂ': 'α', 'ἵ': 'ι', 'ὴ': 'η', 'ὰ': 'α', 'ἅ': 'α', 'ὶ': 'ι', 'ἴ': 'ι',
           'ὸ': 'ο', 'ἥ': 'η', 'ἡ': 'η', 'ὕ': 'υ', 'ἔ': 'ε', 'ἳ': 'ι', 'ὗ': 'υ', 'ἃ': 'α',
           'ὃ': 'ο', 'ὥ': 'ω', 'ὔ': 'υ', 'ῖ': 'ι', 'ἣ': 'η', 'ἷ': 'ι', 'ἑ': 'ε', 'ᾧ': 'ω',
           'ἢ': 'η', 'ΐ': 'ι', }

    rep = dict((nltk.re.escape(k), v) for k, v in rep.items())
    pattern = nltk.re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[nltk.re.escape(m.group(0))], text)

    return text

def remove_final_s(text):
    """Helper function which removes final s characters. Returns a string."""
    return re.sub('ς', '', text)

################### NLP ###################


def count_words(df, column='tokens', process=None, min_freq=2):
    """
    Transform the counter into a Pandas DataFrame with the following function:
    The tokens make up the index of the DataFrame, while the frequency values are stored in a column named freq. 
    The rows are sorted so that the most frequent words appear at the head.
    The last parameter of count_words defines a minimum frequency of tokens to be included in the result. 
    Its default is set to 2 to cut down on tokens occurring only once.
    """
    pipeline = []
    # create counter and run through all data
    counter = Counter()
    # process tokens and update counter

    def update(text):
        tokens = text if process is None else process(text, pipeline=pipeline)
        counter.update(tokens)

    df[column].map(update)
    # transform counter into a DataFrame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'
    return freq_df.sort_values('freq', ascending=False)


def kwic(doc_series, keyword, window=100, print_samples=5):
    """
    The function iteratively collects the keyword contexts by applying the add_kwic function to each document with map. 
    By default, the function returns a list of tuples of the form (left context, keyword, right context). 
    If print_samples is greater than 0, a random sample of the results is printed. 
    Sampling is especially useful with lots of documents because the first entries of the list 
    would otherwise stem from a single or a very small number of documents.
    """
    def add_kwic(text):
        kwic_list.extend(keyword_in_context(
            text, keyword, ignore_case=True, window_width=window))

    kwic_list = []
    doc_series.map(add_kwic)

    if print_samples is None or print_samples == 0:
        return kwic_list
    else:
        k = min(print_samples, len(kwic_list))
        print(f'{k} random samples out of {len(kwic_list)} ' +
              f"contexts for '{keyword}':")
        for sample in random.sample(list(kwic_list), k):
            print(re.sub(r'[\n\t]', ' ', sample[0]) + ' ' +
                  sample[1]+' ' +
                  re.sub(r'[\n\t]', ' ', sample[2]))


def sort_coo(coo_matrix):
    """Sorts the values in the vector while preserving the column index"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_top_n_from_vector(feature_names, sorted_items, topn=10):
    """Get the feature names and tf-idf score of top n items"""
    # Use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # Keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(fname)

    # Create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

################### Visualizations ###################


def display_topics(model, features, num_top_words=5):
    """Utility function to display topics produced by Topic Modelling"""
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]  # inverts sort order
        print("\nTopic %02d" % topic)
        for i in range(0, num_top_words):
            print(" %s (%2.2f)" %
                  (features[largest[i]], word_vector[largest[i]]*100.0/total))


def wordcloud_topics(model, features, no_top_words=40):
    """Utility function to produce wordclouds from Topic Modelling algorithms"""
    for topics, words in enumerate(model.components_):
        size = {}
        largest = words.argsort()[::-1]  # inverts order
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color="black",
                       max_words=100, width=960, height=540)
        wc.generate_from_frequencies(size)
        plt.figure(figsize=(12, 12))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')


def wordcloud_clusters(model, vectors, features, no_top_words=40):
    """Utility function to visualise wordclouds for Kmeans topic modelling"""
    for cluster in np.unique(model.labels_):
        size = {}
        words = vectors[model.labels_ == cluster].sum(axis=0).A[0]
        largest = words.argsort()[::-1]  # invert sort order
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color="black",
                       max_words=100, width=960, height=540)
        wc.generate_from_frequencies(size)
        plt.figure(figsize=(12, 12))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
