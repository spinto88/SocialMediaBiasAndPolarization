#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:28:43 2022

@author: Sofia Morena del Pozo

"""
from pysentimiento import create_analyzer
import unidecode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def is_in(text, keywords, match_type='OR'):
    """
    Check if any of the keywords are present in the given text.

    Parameters:
        text (str): The text to search for keywords.
        keywords (list): List of keywords to search for.
        match_type (str): 'OR' for any keyword match, 'AND' for all keyword match.

    Returns:
        bool: True if there is a match according to match_type, False otherwise.
    """
    matching_keywords = len(set(keywords) & set(text.split(' ')))
    if match_type == 'OR':
        return matching_keywords > 0
    elif match_type == 'AND':
        return matching_keywords == len(keywords)

def SB_Albanese2020(pos_G1, neg_G1, pos_G2, neg_G2, N_G1, N_G2):
    """
    Calculate the Sentiment Bias using the Albanese 2020 formula.

    Parameters:
        pos_G1 (int): Number of positive terms in group 1.
        neg_G1 (int): Number of negative terms in group 1.
        pos_G2 (int): Number of positive terms in group 2.
        neg_G2 (int): Number of negative terms in group 2.
        N_G1 (int): Total number of terms in group 1.
        N_G2 (int): Total number of terms in group 2.

    Returns:
        float: The calculated Sentiment Bias value.
    """
    ans = (pos_G2 - neg_G2) - (pos_G1 - neg_G1)
    norm = N_G1 + N_G2
    if norm != 0:
        return ans / norm
    else:
        return np.nan

def count_terms(text, analyzer, keywords, separator='.'):
    """
    Count the number of positive and negative terms in a text using a sentiment analyzer.

    Parameters:
        text (str): The text to analyze.
        analyzer: Sentiment analyzer.
        keywords (list): List of keywords to count.
        separator (str): Separator used to split the text into terms.

    Returns:
        list: A list containing the number of positive terms, negative terms, and total terms.
    """
    N = 0
    neg = 0
    pos = 0
    for term in text.split(separator):
        if is_in(term, keywords):
            N += 1
            output = analyzer.predict(term).output
            if output == 'POS':
                pos += 1
            elif output == 'NEG':
                neg += 1
    return [pos, neg, N]

def corpus2SB(corpus, group1_keywords,group2_keywords):
    """
    Calculate Sentiment Bias (SB) values for a corpus of texts.

    Parameters:
        corpus (list of str): List of news articles texts.
        group1_keywords (list of str): List of keywords from group 1 to count in each sentence of each corpus.
        group2_keywords (list of str): List of keywords from group 2 to count in each sentence of each corpus.

    Returns:
        list: A list of Sentiment Bias values per outlet.
    """
    news_SB = []
    posG1 = []
    negG1 = []
    posG2 = []
    negG2 = []
    NG2 = []
    NG1 = []

    for news in corpus:
        news = str(news)
        text = unidecode.unidecode(news.lower())
        [pos_G1, neg_G1, N_G1] = count_terms(text, analyzer, group1_keywords)
        [pos_G2, neg_G2, N_G2] = count_terms(text, analyzer, group2_keywords)
        posG1.append(pos_G1)
        negG1.append(neg_G1)
        NG1.append(N_G1)
        posG2.append(pos_G2)
        negG2.append(neg_G2)
        NG2.append(N_G2)
        SB = SB_Albanese2020(pos_G2, neg_G2, pos_G1, neg_G1, N_G2, N_G1)
        news_SB.append(SB)
    return [news_SB, posG1, negG1, NG1, posG2, negG2, NG2]

filename = 'news_articles_corpus.csv'
corpus = pd.read_csv(filename)
corpus = corpus.drop_duplicates()
analyzer = create_analyzer(task="sentiment", lang="es")

# Candidates for President and Vice President of 2019 Argentine national election
G1_keywords = 'cristina,fernandez,alberto,kirchner'.split(',') # FF: Fernandez - Fernandez coalition
G2_keywords = 'macri,pichetto,mauricio'.split(',')  # MP: Macri- Pichetto coalition

text_col = 'body'  
outlet_col = 'outlet'
# Create list of urls, media outlets and the text of the articles to analyze
urls = list(corpus.url)
outlets = list(corpus[outlet_col])
text_articles = list(corpus[text_col])

[news_SB, pos_G1, neg_G1, N_G1, pos_G2, neg_G2, N_G2] = corpus2SB(text_articles, G1_keywords, G2_keywords)

SB_df = pd.DataFrame({'url': urls, 'outlet': outlets, 'SB_FF-MP': news_SB, 'pos_G1': pos_G1, 'neg_G1': neg_G1, 'N_G1': N_G1, 'pos_G2': pos_G2, 'neg_G2': neg_G2, 'N_G2': N_G2})

# In the case of our work, group 1 is FF (Fernandez-Fernandez) and group 1 is MP (Macri-Pichetto)), and the following variables are posff, negff, N_ff, posmp, negmp and N_mp.
corpus['SB'] = news_SB
corpus['pos_G1'] = pos_G1
corpus['neg_G1'] = neg_G1
corpus['N_G1'] = N_G1
corpus['pos_G2'] = pos_G2
corpus['neg_G2'] = neg_G2
corpus['N_G2'] = N_G2

corpus.to_csv('news_articles_corpus_withSB.csv',index = False)