#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:06:07 2022

@author: Sofia Morena del Pozo

source: https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/Multilingual/Spanish/01-Preprocessing-Spanish.html
"""
import pandas as pd
import spacy
import es_core_news_md
nlp = es_core_news_md.load()

path_file = 'news_articles_corpus.csv'
text_column = 'body'
df = pd.read_csv(path_file)
texts = df[str(text_column)].copy().to_list()
texts = list(dict.fromkeys(texts))
texts = texts

# Sacar los numeros


# Create a lemmatized version of the original text file
def lemmatize_text(text):
    document = nlp(text)
    text_lem = ''
    for token in document:
        p = token.lemma_.lower()
        text_lem = text_lem + ' ' + str(p)
    return text_lem

def remove_symbols(text):
    import re
    text = ''.join([i for i in text if not i.isdigit()])
    return re.sub(r'[^\w]', ' ', text)

texts_lem = []
for text in texts:
    t = remove_symbols(text)
    t = lemmatize_text(t)
    texts_lem.append(t)

df['lemmatized_'+str(text_column)] = texts_lem

df.to_csv('lemmatized_data.csv',index = False)


