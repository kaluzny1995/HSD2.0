import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os

from ..extension.lemm import lemmatize_text
from ..constants import (DUPLICATED_PATH, LEMMAS_PATH, HATEFUL_LEMM_DIR, VULGARS_LEMM_DIR,
                         LABELS_SMALL, LABELS_V_SMALL)


def load_lemmatized_tweets(tweets_path=DUPLICATED_PATH, lemmatized_path=LEMMAS_PATH):
    df = pd.read_csv(tweets_path)
    df = df[['id', 'tweet']]

    if not os.path.exists(lemmatized_path):
        df['lemmatized'] = list([lemmatize_text(tweet) for tweet in tqdm(df['tweet'])])
        df[['id', 'lemmatized']].to_csv(lemmatized_path, index=False)
    else:
        df['lemmatized'] = pd.read_csv(lemmatized_path)['lemmatized']

    return df


def load_lemm_phrases(load_vulg=False):
    aphr = list([])
    for label in LABELS_SMALL:
        with open(HATEFUL_LEMM_DIR.replace('{}', label), 'r') as f:
            aphr.append(np.array(f.read().split(';')))
    if load_vulg:
        with open(VULGARS_LEMM_DIR.replace('{}', LABELS_V_SMALL[-1]), 'r') as f:
            aphr.append(np.array(f.read().split(';')))

    return np.array(aphr)
