#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:30:13 2023

@author: Sofia Morena del Pozo
"""
import pandas as pd
import numpy as np

# Definition of the topic's weight in each group
def averageTopicInterest(data, topic, group):
    """
    Calculate the average topic interest in a given group.

    Args:
        data (DataFrame): The input data containing topic and partisan information.
        topic (str): The name of the topic column. We work with 10 topics, so topic takes values between 0 to 9
        group (str): The name of the group column. It can be FF or MP (Center-Left of Center-Right leaning)

    Returns:
        float: The weighted average of the topic's interest in the group.
    """
    return np.average(data[f'T{topic}'], weights=data[f'#{group}'])



def groupByOutlet(df_m):
    """
    Group the data by the outlet media and calculate differences in percentages.

    Args:
        df_m (DataFrame): The input DataFrame with data grouped by outlet.

    Returns:
        DataFrame: The grouped data with a 'difference' column, sorted by the difference in descending order.
    """
    # Group the data by outlet
    data_grouped_by_outlet = df_m.groupby('outlet').sum()

    # Rename columns for consistency
    if 'FF' in data_grouped_by_outlet.columns:
        data_grouped_by_outlet = data_grouped_by_outlet.rename(columns={'FF': '#FF'})
    if 'MP' in data_grouped_by_outlet.columns:
        data_grouped_by_outlet = data_grouped_by_outlet.rename(columns={'MP': '#MP'})

    # Calculate the difference between percentages to sort the data
    difference = (data_grouped_by_outlet['#FF'] / data_grouped_by_outlet['#FF'].sum()) - (data_grouped_by_outlet['#MP'] / data_grouped_by_outlet['#MP'].sum())

    # Add the 'difference' column and sort the data
    data_grouped_by_outlet['difference'] = difference
    data_grouped_by_outlet.sort_values(by='difference', inplace=True, ascending=False)

    return data_grouped_by_outlet

    
# Define the SB (Sentiment Bias) of a news article x. If there are no mentions of a candidate, SB is undefined.
def SB_Albanese2020(x):
    """
    Calculate the Sentiment Bias (SB) of a news item.

    Args:
        x (Series): A row (news item) from the input data with sentiment-related columns.

    Returns:
        float or NaN: The calculated SB if there are mentions, or NaN if no mentions are present.
    """
    ans = (x['pos_CR'] - x['neg_CR']) - (x['pos_CL'] - x['neg_CL']) 
    norm = x['N_CL'] + x['N_CR']
    if norm != 0:
        return ans / norm
    else:
        return np.nan
    
def one_hot_encode(df, col='opinion'):
    """
    Encode a categorical column into one-hot columns with 1s and 0s.

    Args:
        df (DataFrame): The input DataFrame.
        col (str): The name of the column to be one-hot encoded.

    Returns:
        DataFrame: The input DataFrame with one-hot encoded columns.
    """
    one_hot = pd.get_dummies(df[str(col)])
    df = df.join(one_hot)
    return df

def add_accents(x):
    """
    Add accents to specific words in the input string.

    Args:
        x (str): The input string to add accents to.

    Returns:
        str: The input string with added accents.
    """
    x = x.replace('Pagina 12', 'Página 12').replace('Clarin', 'Clarín').replace('La Nacion', 'La Nación').replace('El Dia', 'El Día')
    return x

def bootstraping(dist,sample_n,n_bootst = 10000):
    dist_boots = []
    for i in range(n_bootst):
        #dist_boots.append(np.mean(dist.sample(n = dist.shape[0],replace = True)))
        dist_boots.append(np.mean(dist.sample(n = sample_n,replace = True)))
    return dist_boots


def index_sorted_list(l):
    return sorted(range(len(l)), key=lambda k: l[k])