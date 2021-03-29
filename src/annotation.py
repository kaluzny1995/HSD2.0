import numpy as np
import pandas as pd

import warnings
import krippendorff
from sklearn.metrics import cohen_kappa_score

from src.dataframes.utils import shuffle_dataframe
from src.dataframes.utils import classes
from src.dataframes.cards import class_single_cards
from src.constants import LABELS, ANNOTATION_SHEET_PATH


def annotation_sheet(df, min_class=50, shuffle=False):
    if shuffle:
        df = shuffle_dataframe(df)

    df_c = classes(df, convert_null=True)
    df_sc = class_single_cards(df_c)
    cards = df_sc[['cardinality']].T[LABELS].values.flatten()
    if not all([c >= min_class for c in cards]):
        warnings.warn(f'Not all class cardinalities are bigger than or equal to given minimum class cardinality!',
                      category=RuntimeWarning, stacklevel=1)

    idx, interesting_ids = 0, list([])
    current_cards = np.zeros(len(cards), dtype=np.int32)
    while len(interesting_ids) < len(cards) * min_class:
        vals = df_c.iloc[idx][LABELS].values
        incomplete = list([cc < min(min_class, c) for cc, c in zip(current_cards, cards)])
        if any(list([ic and bool(v) for ic, v in zip(incomplete, vals)])):
            interesting_ids.append(idx)

        idx += 1

    df_interest = df.iloc[interesting_ids][['id', 'tweet'] + LABELS]
    df_interest_empty = pd.DataFrame(df_interest.values, columns=df_interest.columns)
    clear = np.empty(df_interest_empty[LABELS].values.shape)
    clear[:] = np.nan
    df_interest_empty[LABELS] = clear

    df_interest.to_csv(ANNOTATION_SHEET_PATH.replace('{}', ''), index=False)
    df_interest_empty.to_csv(ANNOTATION_SHEET_PATH.replace('{}', '_empty'), index=False)

    return df_interest


def annotation_agreement(default_annotator='a0', annotators=list(['a1'])):
    df_annotated_main = pd.read_csv(ANNOTATION_SHEET_PATH.replace('{}', f'_{default_annotator}'))
    labels_main = np.array(df_annotated_main[LABELS].fillna(0.).values, dtype=np.int).flatten()

    ka, ck = list([]), list([])
    for annotator in annotators:
        df_annotated = pd.read_csv(ANNOTATION_SHEET_PATH.replace('{}', f'_{annotator}'))
        labels = np.array(df_annotated[LABELS].fillna(0.).values, dtype=np.int).flatten()
        ka.append(krippendorff.alpha(reliability_data=[labels_main, labels]))
        ck.append(cohen_kappa_score(y1=labels_main, y2=labels))

    return np.array(ka), np.array(ck)