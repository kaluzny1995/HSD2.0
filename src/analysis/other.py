import csv

from tqdm.notebook import tqdm

from ..utils.texts import text_sentiment, text_numbers
from ..constants import OTHER_SCORES_PATH


def analyse_other(df):
    csv_labels = list([
        'id',
        's_neg', 's_neu', 's_pos',
        'n_chars', 'n_sylls', 'n_words', 'nu_words',
        'nl_chars', 'nl_sylls', 'nl_words', 'nlu_words',
    ])
    with open(OTHER_SCORES_PATH, 'w') as f:
        csv.writer(f).writerow(csv_labels)
    del csv_labels

    for _, tweet in tqdm(df.iterrows(), total=len(df)):
        scores = dict({})

        scores['neg'], scores['neu'], scores['pos'] = text_sentiment(tweet['tweet'])
        scores['chars'], scores['sylls'], scores['words'], scores['u_words'] = text_numbers(tweet['tweet'])
        scores['l_chars'], scores['l_sylls'], scores['l_words'], scores['l_u_words'] = text_numbers(tweet['lemmatized'])

        csv_values = list([
            tweet['id'],
            scores['neg'], scores['neu'], scores['pos'],
            scores['chars'], scores['sylls'], scores['words'], scores['u_words'],
            scores['l_chars'], scores['l_sylls'], scores['l_words'], scores['l_u_words']
        ])
        with open(OTHER_SCORES_PATH, 'a') as f:
            csv.writer(f).writerow(csv_values)
        del scores, csv_values


def get_other(text_tweet, lemm_tweet):
    scores = list([])

    scores.extend(list(text_sentiment(text_tweet)))
    scores.extend(list(text_numbers(text_tweet)))
    scores.extend(list(text_numbers(lemm_tweet)))

    return scores
