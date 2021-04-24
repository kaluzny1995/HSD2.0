import numpy as np
import pandas as pd

import string
from src.dataframes.utils import combine_row_wisely
from src.constants import LABELS, POLISH_STOPWORDS


def class_predictions(y_true, y_pred, y_prob=None, cls='wyzywanie'):
    y_t = pd.DataFrame(y_true, columns=LABELS)[cls].values
    y_p = pd.DataFrame(y_pred, columns=LABELS)[cls].values
    y_pb = None if y_prob is None else pd.DataFrame(y_prob, columns=LABELS)[cls].values

    return y_t, y_p, y_pb


def split_error_types(y_true, y_pred, y_prob=None, additional_data=None, cls='wyzywanie'):
    y_t, y_p, y_pb = class_predictions(y_true=y_true, y_pred=y_pred, y_prob=y_prob, cls=cls)

    df = pd.DataFrame({
        'ground truth': y_t,
        'prediction': y_p
    }).astype(int)
    if y_prob is not None:
        df['probability'] = y_pb
        df['error'] = df.apply(lambda x: np.abs(x['ground truth'] - x['probability']), axis=1)
        df = df.sort_values('error', ascending=False)
    if additional_data is not None:
        cols = range(len(additional_data.shape[1])) if type(
            additional_data) != pd.DataFrame else additional_data.columns
        df = combine_row_wisely([pd.DataFrame(additional_data, columns=cols), df])

    if 'lemmatized' in df.columns:
        df['lemmatized'] = df.apply(lambda x: x['lemmatized'].translate(str.maketrans('', '', string.punctuation)),
                                    axis=1)

    fn = df[(df['ground truth'] == 1) & (df['prediction'] == 0)]
    fp = df[(df['ground truth'] == 0) & (df['prediction'] == 1)]
    tn = df[(df['ground truth'] == 0) & (df['prediction'] == 0)]
    tp = df[(df['ground truth'] == 1) & (df['prediction'] == 1)]

    return fn, fp, tn, tp


def most_common_word(texts, min_count=1, stopwords=POLISH_STOPWORDS):
    text = ' '.join(texts)
    words_dict = dict({})

    for word in text.split(' '):
        if word not in stopwords:
            if word not in words_dict:
                words_dict[word] = 0
            words_dict[word] += 1

    df = pd.DataFrame({
        'word': words_dict.keys(),
        'count': words_dict.values()
    }).sort_values(['count', 'word'], ascending=[False, True])
    df = df[df['count'] >= min_count]

    return df


def most_common_phrase(texts, length=3, min_count=1, stopwords=POLISH_STOPWORDS):
    phrases_dict = dict({})

    for text in texts:
        words = list([word for word in text.split(' ') if word not in stopwords])
        if len(words) - length >= 0:
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                if phrase not in phrases_dict:
                    phrases_dict[phrase] = 0
                phrases_dict[phrase] += 1

    df = pd.DataFrame({
        'phrase': phrases_dict.keys(),
        'count': phrases_dict.values()
    }).sort_values(['count', 'phrase'], ascending=[False, True])
    df = df[df['count'] >= min_count]

    return df


def hashtags(df):
    dff = df[['hashtags', 'error']]
    dff['hashtags'] = dff.apply(lambda x: list([ht[1:-1] for ht in x['hashtags'][1:-1].split(', ')]), axis=1)
    dff = dff[df.astype(str)['hashtags'] != '[]'].rename(columns={'hashtags': 'hashtag'})

    dff = dff.explode('hashtag')
    dff = dff.groupby('hashtag')['error'].apply(lambda x: np.square(x).mean()).reset_index()
    dff = dff.sort_values(['error', 'hashtag'], ascending=[False, True])

    return dff


def usernames(df):
    dff = df[['username', 'error']]
    dff = dff.groupby('username')['error'].apply(lambda x: np.square(x).mean()).reset_index()
    dff = dff.sort_values(['error', 'username'], ascending=[False, True])

    return dff
