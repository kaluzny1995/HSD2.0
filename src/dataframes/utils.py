import csv
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        weigths = np.array([2 - c/len(df) for c in srs_c.values])
    else:
        weigths = np.ones(len(LABELS))

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


def models_quality_results(y_trues, y_preds_s, model_names, additionals=None, additional_titles=None, save_file=None):
    if additionals and additional_titles:
        assert len(additionals) == len(additional_titles), 'Additional data must have the same length as titles!'

    def adapt(y):
        y = np.array([[yy.T[i] for yy in y] for i in range(y.shape[2])])
        y_all = np.array([np.concatenate([yy[i] for yy in y]) for i in range(y.shape[1])])
        y = list([*y, y_all])

        return y

    labels = LABELS + ['overall']
    y_trues_s = np.array([y_trues for _ in range(len(model_names))])
    y_trues_s = adapt(y_trues_s)
    y_preds_s = adapt(y_preds_s)

    df = pd.DataFrame({
        'model': model_names
    })

    for i, (label, y_true, y_pred) in enumerate(zip(labels, y_trues_s, y_preds_s)):

        df[f'{label}_A'] = np.array([accuracy_score(y_true=y_t, y_pred=y_p) for y_t, y_p in zip(y_true, y_pred)])
        measure_fns = list([precision_score, recall_score, f1_score])
        for measure_fn, measure_l in zip(measure_fns, ['P', 'R', 'F']):
            m0, m1 = np.array([measure_fn(y_true=y_t, y_pred=y_p, labels=[0, 1], average=None, zero_division=1.)
                               for y_t, y_p in zip(y_true, y_pred)]).T
            df[f'{label}_{measure_l}0'] = m0
            df[f'{label}_{measure_l}1'] = m1

    if additionals:
        add_titles = list([f'Add.: {i+1}' for i in range(len(additionals))])\
            if not additional_titles else additional_titles
        df_add = pd.DataFrame(np.array(additionals).T, columns=add_titles) #chk
        df = combine_row_wisely([df, df_add])

    if save_file:
        df.to_csv(save_file, index=False)
    df = df.set_index('model')

    return df
