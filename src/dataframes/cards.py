import numpy as np
import pandas as pd

from ..constants import LABELS


def class_single_cards(df_c):
    df = df_c[df_c.columns]

    df['total'] = np.ones(len(df_c), dtype=np.int32)
    srs_c = df.sum().sort_values(ascending=False)
    srs_perc = pd.Series(srs_c / len(df) * 100).sort_values(ascending=False)

    df_sc = pd.DataFrame({
        'label': srs_c.index,
        'cardinality': srs_c.values,
        '%': srs_perc.values
    })
    df_sc = df_sc.set_index('label')

    return df_sc


def class_single_cards_comp(df_sc_c, df_sc_d):
    df_comp = pd.DataFrame({
        'hate type': df_sc_c.index,
        'card. before': df_sc_c['cardinality'].values,
        '% before': df_sc_c['%'].values,
        'card. after': df_sc_d['cardinality'].values,
        '% after': df_sc_d['%'].values,
    })
    df_comp = df_comp.set_index('hate type')

    return df_comp


def class_combination_cards(df_c):
    df = df_c[df_c.columns]

    df['cardinality'] = np.ones(len(df_c), dtype=np.int32)
    df_cc = df.groupby(LABELS).count().sort_values(by='cardinality', ascending=False)
    df_cc['%'] = df_cc['cardinality'] / len(df_c) * 100

    return df_cc


def phrase_cards(aphr):
    df_pc = pd.DataFrame({
        'hate type': LABELS,
        'cardinality': [len(phr) for phr in aphr]
    })
    df_pc = df_pc.set_index(['hate type'])

    return df_pc


def phrase_cards_comp(lemm_aphr, ext_aphr):
    df_pcc = pd.DataFrame({
        'hate type': LABELS,
        'cardinality': [len(phr) for phr in lemm_aphr],
        # 'ext. count': [len(phr) for phr in ext_aphr]
        'ext. cardinality': [380, 400, 250, 200, 750, 900, 50]
    })
    df_pcc = df_pcc.set_index(['hate type'])

    return df_pcc
