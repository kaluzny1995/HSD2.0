import numpy as np
import itertools

from .extension.lemm import lemmatize_text


def POC(text, phrase, lemmatized=False, stopwords=[]):

    t = text if lemmatized else lemmatize_text(text)
    p = phrase if lemmatized else lemmatize_text(phrase)

    t_words = list(filter(lambda x: x not in stopwords, t.split(' ')))
    p_words = list(filter(lambda x: x not in stopwords, p.split(' ')))

    assert (len(t_words) > 0), 'The examined text must have at least one non-stopword word!'

    if len(p_words) > 1:
        occurences = list([[i for i, x in enumerate(t_words) if x == p_w] for p_w in p_words])
        occurences = list([o for o in occurences if len(o) > 0])

        orders = list(itertools.product(*occurences))
        order_pairs_list = list([[tuple((o[i], o[ i +1])) for i, oi in enumerate(o[:-1])] for o in orders])

        coeffs = list([sum([1. if op[0 ] <op[1] else -1. for op in ops] ) /(len(p_words) - 1)
                       for ops in order_pairs_list])

        return (np.min(coeffs), np.mean(coeffs), np.max(coeffs))
    elif len(p_words) == 1:
        return (1., 1., 1.) if p_words[0] in t_words else (0., 0., 0.)
    else:
        return (0., 0., 0.)