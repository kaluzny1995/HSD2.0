import numpy as np
import warnings

import pickle

from ..vectorizers.Vectorizer import Vectorizer
from ..constants import CHV_MODEL_DIR


class CharacterVectorizer(Vectorizer):
    def __init__(self, name='CharacterVectorizer', length=100):
        super(CharacterVectorizer, self).__init__(name=name)
        self._alphabet = [chr(i) for i in range(1000)]
        self.length = length

    def __str__(self):
        return self.name

    def fit(self, X):
        super().fit(X)

    def _vectorize(self, texts):
        vectors = list([])
        for text in texts:
            text_vector = np.array([0. if ch not in self._alphabet else self._alphabet.index(ch) + 1. for ch in text])
            if len(text) > self.length:
                warnings.warn(f'Desired length of vector greater than vector length. Excessive text will be truncated!',
                              category=RuntimeWarning, stacklevel=1)
                vectors.append(text_vector[:self.length])
            elif len(text_vector) < self.length:
                vectors.append(np.concatenate([text_vector, np.zeros(self.length - len(text_vector))]))
            else:
                vectors.append(text_vector)

        vectors = np.array(vectors) / len(self._alphabet)
        
        return vectors

    def transform(self, X):
        super().transform(X)
        return self._vectorize(X)

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)
        return self._vectorize(X)

    def save(self, save_file=CHV_MODEL_DIR):
        super().save(save_file)
        model_dict = dict({
            'name': self.name
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=CHV_MODEL_DIR):
        super().load(load_file)
        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
