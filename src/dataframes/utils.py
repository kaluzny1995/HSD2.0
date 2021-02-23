import csv
import pandas as pd

from ..constants import LABELS


def classes(df, convert_null=False):
    df_c = df[LABELS]
    if convert_null:
        df_c = df_c.notnull().astype('int')

    return df_c


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
