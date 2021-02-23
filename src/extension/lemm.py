import numpy as np
from combo.predict import COMBO

from tqdm.notebook import tqdm

from ..constants import TAGGER_MODEL

tagger = COMBO.from_pretrained(TAGGER_MODEL)


def lemmatize_text(text):
    text = text.replace('#', '').replace('[...]', '')
    sentence = tagger(text)

    lemmas = [token.lemma.lower() for token in sentence.tokens if token.deprel != 'punct']
    lemm_text = ' '.join(lemmas)

    return lemm_text


def get_lemmatized_phrases(phrases, save_file=None):
    lemm_phrases = list([])

    for phrase in tqdm(phrases, leave=None):
        lemm_phrase = lemmatize_text(phrase)
        if lemm_phrase not in lemm_phrases:
            lemm_phrases.append(lemm_phrase)

    if save_file:
        with open(save_file, 'w') as f:
            f.write(';'.join(lemm_phrases))

    return np.array(lemm_phrases)
