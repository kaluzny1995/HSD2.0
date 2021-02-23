import numpy as np
import csv
import pickle

from tqdm.notebook import tqdm

from .lda import lda_topics
from ..measures import POC
from ..constants import LABELS_V_SMALL, TOPIC_POC_SCORES_PATH, LDA_MODEL_DIR, POLISH_STOPWORDS


def analyse_topic_POC(df, n_words=10):
    csv_labels = list(['id'])
    for label in LABELS_V_SMALL:
        csv_labels.append(f'{label}_topic_POC_min')
        csv_labels.append(f'{label}_topic_POC_mean')
        csv_labels.append(f'{label}_topic_POC_max')
    with open(TOPIC_POC_SCORES_PATH, 'w') as f:
        csv.writer(f).writerow(csv_labels)
    del csv_labels

    for _, tweet in tqdm(df.iterrows(), total=len(df)):
        scores = dict({})

        for label in LABELS_V_SMALL:
            with open(LDA_MODEL_DIR.replace('{}', label), 'rb') as f:
                lda_model, cv = pickle.load(f)

            topics = lda_topics(lda_model, cv, n_words=n_words)
            sc_min, sc_mean, sc_max = list([]), list([]), list([])

            for topic in topics:
                mn, mean, mx = POC(tweet['lemmatized'], topic, lemmatized=True, stopwords=POLISH_STOPWORDS)
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
        with open(TOPIC_POC_SCORES_PATH, 'a') as f:
            csv.writer(f).writerow(csv_values)
        del scores, csv_values


def get_topic_POC(lemm_tweet, n_words=10):
    scores = list([])

    for label in LABELS_V_SMALL:
        with open(LDA_MODEL_DIR.replace('{}', label), 'rb') as f:
            lda_model, cv = pickle.load(f)

        topics = lda_topics(lda_model, cv, n_words=n_words)
        sc_min, sc_mean, sc_max = list([]), list([]), list([])

        for topic in topics:
            mn, mean, mx = POC(lemm_tweet, topic, lemmatized=True, stopwords=POLISH_STOPWORDS)
            sc_min.append(mn)
            sc_mean.append(mean)
            sc_max.append(mx)

        scores.append(np.min(sc_min))
        scores.append(np.mean(sc_mean))
        scores.append(np.max(sc_max))

        del sc_min, sc_mean, sc_max

    return scores
