import numpy as np


# morphological closing operation of two-values list
# (p - positive element | n - negative element | threshold - min. empty space length)
def closing_empty_spaces(values, p=.0, n=None, threshold=10):
    es_b, positive = list([]), n
    for i, es in enumerate(values):
        # if currently is positive but previously was not
        if es == p and positive == n:
            es_b.append(i)
            positive = p
        # if currently is negative but previously was
        if es == n and positive == p:
            es_b.append(i)
            positive = n

    # if length of empty bound is less than threshold then remove it
    e = list(zip(es_b[::2], es_b[1::2]))
    e_accepted = list([])
    for e0, e1 in e:
        if e1-e0 < threshold:
            for i in range(e0, e1):
                values[i] = None
        else:
            e_accepted.append((e0, e1))
    return values, e_accepted


def union3(s0, s1, s2):
    s = np.union1d(s0, s1)
    s = np.union1d(s, s2)
    s = s[s != '']

    return s
