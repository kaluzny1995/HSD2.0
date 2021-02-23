import numpy as np


def combine_phrases(lemmatized, similar, synonymic, save_file=None):
    all_phrases = np.union1d(similar, synonymic)
    all_phrases = np.union1d(lemmatized, all_phrases)
    all_phrases = all_phrases[all_phrases != '']

    if save_file:
        with open(save_file, 'w') as f:
            f.write(';'.join(all_phrases))

    return all_phrases
