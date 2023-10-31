#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:15:26 2022

@author: Sofia Morena del Pozo

"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import nltk
nltk.download('stopwords')

# Library for creating word clouds
from wordcloud import WordCloud

# sklearn objects for topic modeling
from sklearn.feature_extraction.text import CountVectorizer  # Term frequency counter
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF creator

# Topic decomposition algorithms
from sklearn.decomposition import NMF

# Read data from 'news_articles_corpus.csv' into a DataFrame
df = pd.read_csv('news_articles_corpus.csv')
texts = df.lemmatized_body.copy().to_list()
texts = list(dict.fromkeys(texts))  # Remove duplicates

# List of stopwords
stopwords = nltk.corpus.stopwords.words('spanish')

# Create the word count object, specifying to remove
# stopwords, terms that appear in only one document (min_df),
# and terms that appear in more than 70% of the documents (max_df).
# This is to eliminate rare words (or typos) and terms that are
# likely stopwords not included in the list.
count = CountVectorizer(min_df=2, max_df=0.70, stop_words=stopwords, lowercase=True)

# Fit it with the data, specifically creating a document-term matrix
x_count = count.fit_transform(texts)

# Dimensions of the document-term matrix
print(x_count.shape)

# Create the TF-IDF object, specifying to return document vectors
# with Euclidean norm equal to 1 (norm='l2')
tfidf = TfidfTransformer(norm='l2')

# Create the TF-IDF matrix from the frequency matrix
x_tfidf = tfidf.fit_transform(x_count)

# Choose the number of topics
n_components = 10

# Create the NMF object with the specified topics
nmf = NMF(n_components=n_components)

# Apply it to our data
x_nmf = nmf.fit_transform(x_tfidf)

# Dimensions of the transformed matrix
print(x_nmf.shape)

# Create a vocabulary dictionary to map terms to their indices
vocabulary = {item: key for key, item in count.vocabulary_.items()}

# Matrix H = x_nmf
H = x_nmf.copy()

# Matrix W
W = nmf.components_.copy()

# For each component
for n in range(n_components):

    # Sort a list of the vocabulary size by weight in each component and take the top 10
    list_sorted = sorted(range(W.shape[1]), reverse=True, key=lambda x: W[n][x])[:10]

    # Print the terms associated with the largest values in each component
    print('Topic: ', n)
    print(', '.join([vocabulary[i] for i in list_sorted]))
    print('\n')
