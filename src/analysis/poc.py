import numpy as np
import csv

from tqdm.notebook import tqdm

from ..measures import POC
from ..constants import LABELS_V_SMALL, POC_SCORES_PATH, POLISH_STOPWORDS


def analyse_POC(df, phrs, save_file=POC_SCORES_PATH):
    csv_labels = list(['id'])
    for label in LABELS_V_SMALL:
        csv_labels.append(f'{label}_POC_min')
        csv_labels.append(f'{label}_POC_mean')
        csv_labels.append(f'{label}_POC_max')
    with open(save_file, 'w') as f:
        csv.writer(f).writerow(csv_labels)
    del csv_labels

    for _, tweet in tqdm(df.iterrows(), total=len(df)):
        scores = dict({})

        for label, phrases in zip(LABELS_V_SMALL, phrs):
            sc_min, sc_mean, sc_max = list([]), list([]), list([])

            for phrase in phrases:
                mn, mean, mx = POC(tweet['lemmatized'], phrase, lemmatized=True, stopwords=POLISH_STOPWORDS)
                sc_min.append(mn)
                sc_mean.append(mean)
                sc_max.append(mx)

            scores[f'{label}_min'] = np.min(sc_min)
            scores[f'{label}_mean'] = np.mean(sc_mean)
            scores[f'{label}_max'] = np.max(sc_max)
            del sc_min, sc_mean, sc_max

        csv_values = list([tweet['id']])
        for label in LABELS_V_SMALL:
            csv_values.append(scores[f'{label}_min'])
            csv_values.append(scores[f'{label}_mean'])
            csv_values.append(scores[f'{label}_max'])
        with open(save_file, 'a') as f:
            csv.writer(f).writerow(csv_values)
        del scores, csv_values


def get_POC(lemm_tweet, phrs):
    scores = list([])

    for label, phrases in zip(LABELS_V_SMALL, phrs):
        sc_min, sc_mean, sc_max = list([]), list([]), list([])

        for phrase in phrases:
            mn, mean, mx = POC(lemm_tweet, phrase, lemmatized=True, stopwords=POLISH_STOPWORDS)
            sc_min.append(mn)
            sc_mean.append(mean)
            sc_max.append(mx)

        scores.append(np.min(sc_min))
        scores.append(np.mean(sc_mean))
        scores.append(np.max(sc_max))

        del sc_min, sc_mean, sc_max

    return scores
