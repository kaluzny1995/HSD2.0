import numpy as np

from polyglot.text import Text
from polyglot.detect.base import logger as polyglot_logger
import pyphen

from ..constants import HYPHENATION_MODEL

polyglot_logger.setLevel("ERROR")

dic = pyphen.Pyphen(lang=HYPHENATION_MODEL)


def text_sentiment(text):
    # detect and delete invalid characters first
    t = text
    invalid = set()
    for i, ch in enumerate(t):
        try:
            Text(f"Char: {ch}").words
        except:
            invalid.add(ch)
    for ch in invalid:
        t = t.replace(ch, '')

    t = Text(t)
    sents = list([])
    for w in t.words:
        try:
            s = w.polarity
        except ValueError:
            s = 0
        sents.append(s)
    sents = np.array(sents)

    return np.size(sents[sents == -1]), np.size(sents[sents == 0]), np.size(sents[sents == 1])


def text_numbers(text):
    num_chars = len(text.replace(' ', ''))
    num_syllables = sum([len(dic.inserted(word).split('-')) for word in text.split(' ')])
    num_words = len(text.split(' '))
    num_unique_words = len(set(text.lower().split(' ')))

    return num_chars, num_syllables, num_words, num_unique_words
