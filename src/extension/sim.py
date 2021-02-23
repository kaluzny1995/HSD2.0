import numpy as np
import itertools
from scipy.spatial.distance import cosine

from tqdm.notebook import tqdm

from polyglot.text import Text
from polyglot.detect.base import logger as polyglot_logger
import spacy

from .lemm import lemmatize_text, tagger
from ..constants import SPACY_PL_MODEL

polyglot_logger.setLevel("ERROR")

pl_nlp = spacy.load(SPACY_PL_MODEL)


def similar_phrases(phrase, sim_threshold=0.6, lemmatized=True, stopwords=[], negative_sentiment_only=True):
    p = phrase if lemmatized else lemmatize_text(phrase)

    # detect and delete invalid characters first
    invalid = set()
    for i, ch in enumerate(p):
        try:
            Text(f"Char: {ch}").words
        except:
            invalid.add(ch)
    for ch in invalid:
        p = p.replace(ch, '')

    wordtokens = list([wt for wt in pl_nlp(str(p)) if wt.text not in stopwords and wt.has_vector])
    cosine_similarity = lambda x, y: 1 - cosine(x, y)

    similar_phrases_options = list([])
    for wordtoken in tqdm(wordtokens, leave=None):
        wordtoken_options = list([])
        words = list([word for word in pl_nlp.vocab if word.has_vector and '.' not in word.text])
        for word in tqdm(words, leave=None):
            try:
                sentiment = Text(word.text).words[0].polarity
            except Exception:
                sentiment = 0

            lemma = tagger(word.text).tokens[0].lemma.lower()
            sim = cosine_similarity(wordtoken.vector, word.vector)

            if (not negative_sentiment_only or sentiment == -1) \
                    and sim >= sim_threshold and lemma not in wordtoken_options:
                wordtoken_options.append(lemma)

        if len(wordtoken_options) == 0:
            wordtoken_options.append(wordtoken.text)
        similar_phrases_options.append(wordtoken_options)

    options = list(itertools.product(*similar_phrases_options))
    options = list([' '.join(option) for option in options])
    if p in options:
        options.remove(p)

    return options


def get_similar_phrases(phrases, lemmatized=True, stopwords=[], save_file=None):
    all_sim_phrases = list([])

    t = tqdm(phrases, leave=None)
    for phrase in t:
        sim_phrases = similar_phrases(phrase, lemmatized=lemmatized, stopwords=stopwords)
        for sp in sim_phrases:
            if sp not in all_sim_phrases:
                all_sim_phrases.append(sp)
        t.set_postfix_str(f'Found so far: {len(all_sim_phrases)}')

    if save_file:
        with open(save_file, 'w') as f:
            f.write(';'.join(all_sim_phrases))

    return np.array(all_sim_phrases)
