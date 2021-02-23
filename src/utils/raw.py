import numpy as np

from ..constants import LABELS_SMALL, LABELS_V_SMALL, HATEFUL_RAW_DIR, VULGARS_RAW_DIR


def save_raw_phrases(df_phr, hate_type, file):
    phr = df_phr[df_phr[hate_type] == 1]['klucze'].values
    with open(file, 'w') as f:
        f.write(';'.join(phr))

    return phr


def load_raw_phrases():
    aphr = list([])
    for label in LABELS_SMALL:
        with open(HATEFUL_RAW_DIR.replace('{}', label), 'r') as f:
            aphr.append(np.array(f.read().split(';')))
    with open(VULGARS_RAW_DIR.replace('{}', LABELS_V_SMALL[-1]), 'r') as f:
        aphr.append(np.array(f.read().split(';')))

    return np.array(aphr)


def save_raw_vulgars(vulgars, file):
    with open(file, 'w') as f:
        f.write(';'.join(vulgars))
