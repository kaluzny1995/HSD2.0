import numpy as np
import warnings

import pickle

from tqdm.notebook import tqdm

from ..vectorizers.Vectorizer import Vectorizer
from ..constants import WSBV_MODEL_DIR


class WordSimpleBoWVectorizer(Vectorizer):
    def __init__(self, name='WordSimpleBoWVectorizer', length=100):
        super(WordSimpleBoWVectorizer, self).__init__(name=name)
        self.n_words = 1
        self._bow = dict({})
        self.frequencies = dict({})
        self.length = length

    def __str__(self):
        return self.name

    def fit(self, X):
        super().fit(X)
        for x in tqdm(X, leave=None):
            words = x.split(' ')
            for word in words:
                if word not in self._bow:
                    self._bow[word] = float(self.n_words)
                    self.frequencies[word] = 0
                    self.n_words += 1
                self.frequencies[word] += 1

    def transform(self, X):
        super().transform(X)
        vectors = list([])
        for x in X:
            text_vector = np.array([0. if word not in self._bow else self._bow[word] for word in x.split(' ')])
            if len(text_vector) > self.length:
                warnings.warn(f'Desired length of vector greater than vector length. Excessive text will be truncated!',
                              category=RuntimeWarning, stacklevel=1)
                vectors.append(text_vector[:self.length])
            elif len(text_vector) < self.length:
                vectors.append(np.concatenate([text_vector, np.zeros(self.length - len(text_vector))]))
            else:
                vectors.append(text_vector)

        vectors = np.array(vectors) / self.n_words

        return vectors

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)
        return self.transform(X)

    def save(self, save_file=WSBV_MODEL_DIR):
        super().save(save_file)
        model_dict = dict({
            'name': self.name,
            'n_words': self.n_words,
            'bow': self._bow,
            'freq': self.frequencies
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=WSBV_MODEL_DIR):
        super().load(load_file)
        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
        self.n_words = model_dict['n_words']
        self._bow = model_dict['bow']
        self.frequencies = model_dict['freq']
