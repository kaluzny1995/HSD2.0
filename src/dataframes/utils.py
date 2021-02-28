import csv
import numpy as np
import pandas as pd

from ..constants import LABELS


def classes(df, convert_null=False):
    df_c = df[LABELS]
    if convert_null:
        df_c = df_c.notnull().astype('int')

    return df_c


def class_weights(df_c, w_type=0):
    assert w_type in range(3), 'Unknown weighting type! Select 0 (1/ class card), 1 (all cards/class card), 2 (1-class card/all card).'
    df = df_c[df_c.columns]

    srs_c = df.sum()
    if w_type == 0:
        weigths = np.array([1/c for c in srs_c.values])
    elif w_type == 1:
        weigths = np.array([len(df)/c for c in srs_c.values])
    elif w_type == 2:
        weigths = np.array([2 - c/srs_c.values.sum() for c in srs_c.values])
    else:
        weigths = np.array([1/c for c in srs_c.values])

    return weigths


def phrases(df_c):
    df_phr = df_c[['klucze'] + LABELS]
    df_phr = df_phr.notnull().astype('int')
    df_phr['klucze'] = df_c['klucze']
    df_phr = df_phr.dropna()
    df_phr['klucze'] = list([phr.replace('[..]', '[...]') for phr in df_phr['klucze']])
    df_phr['klucze'] = list([k.split(';') for k in df_phr['klucze']])
    # convert list-like column elements to separate rows
    df_phr = df_phr.explode('klucze')

    return df_phr


def combine_row_wisely(dfs):
    it = iter(dfs)
    length = len(next(it))
    if not all(len(l) == length for l in it):
        raise ValueError('Not all dataframes have the same length!')

    return pd.concat(dfs, axis=1)


def shuffle_dataframe(df):
    return df.sample(frac=1)


def reduce_to_polish(in_file, out_file):
    with open(in_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        with open(out_file, 'w') as wf:
            writer = csv.writer(wf)
            writer.writerow(header)

            for row in reader:
                if row[11] == 'pl':
                    writer.writerow(row)
