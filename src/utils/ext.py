import numpy as np

from ..constants import LABELS_SMALL, LABELS_V_SMALL, HATEFUL_EXT_DIR, VULGARS_EXT_DIR


def load_ext_phrases(load_vulg=False):
    aphr = list([])
    for label in LABELS_SMALL:
        with open(HATEFUL_EXT_DIR.replace('{}', label), 'r') as f:
            aphr.append(np.array(f.read().split(';')))
    if load_vulg:
        with open(VULGARS_EXT_DIR.replace('{}', LABELS_V_SMALL[-1]), 'r') as f:
            aphr.append(np.array(f.read().split(';')))

    return np.array(aphr)
