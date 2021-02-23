import numpy as np
import itertools
from pyplwnxml import PlwnxmlParser

from tqdm.notebook import tqdm

from ..constants import WORDNET_PATH
from .lemm import lemmatize_text

pl_wordnet = PlwnxmlParser(WORDNET_PATH).read_wordnet()


def synonymic_phrases(phrase, lemmatized=True, stopwords=[], negative_sentiment_only=True):
    synonymic_phrases_options = list([])
    p = phrase if lemmatized else lemmatize_text(phrase)

    lemm_words = list([w for w in p.split(' ') if w not in stopwords])

    for lemm_word in lemm_words:
        lemm_word_options = list([])
        for lemm in pl_wordnet.lemma(lemm_word):
            for synset in lemm.synsets:
                for lu in synset.lexical_units:
                    if (not negative_sentiment_only or lu.sentiment in ['- m', '- s']) \
                            and lu.name not in lemm_word_options:
                        lemm_word_options.append(lu.name)

        if len(lemm_word_options) == 0:
            lemm_word_options.append(lemm_word)
        synonymic_phrases_options.append(lemm_word_options)

    options = list(itertools.product(*synonymic_phrases_options))
    options = list([' '.join(option) for option in options])
    if p in options:
        options.remove(p)

    return options


def get_synonymic_phrases(phrases, lemmatized=True, stopwords=[], save_file=None):
    all_syn_phrases = list([])

    t = tqdm(phrases, leave=None)
    for phrase in t:
        syn_phrases = synonymic_phrases(phrase, lemmatized=lemmatized, stopwords=stopwords)
        for sp in syn_phrases:
            if sp not in all_syn_phrases:
                all_syn_phrases.append(sp)
    t.set_postfix_str(f'Found so far: {len(all_syn_phrases)}')

    if save_file:
        with open(save_file, 'w') as f:
            f.write(';'.join(all_syn_phrases))

    return np.array(all_syn_phrases)
