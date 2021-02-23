import numpy as np
import warnings

from ..vectorizers.Vectorizer import Vectorizer


class CharacterVectorizer(Vectorizer):
    def __init__(self, name='CharacterVectorizer', length=1000):
        super(Vectorizer, self).__init__(name=name)
        self.alphabet = [chr(i) for i in range(1000)]
        self.length = length

    def __str__(self):
        return self.name

    def fit(self):
        super().fit()

    def _vectorize(self, texts):
        vector = list([])
        for text in texts:
            text_vector = np.zeros(np.max([len(text), self.length]), dtype=np.float32)
            if len(text) > self.length:
                warnings.warn(f'Length of text greater than vector length. Excessive text will be truncated!', category='TruncationWarning', stacklevel=2)

            for i, ch in enumerate(text):
                text_vector[i] = 0. if ch not in self.alphabet else self.alphabet.index(ch) + 1.
            vector.append(text_vector[:self.length])

        vector = np.array(vector)/len(self.alphabet)
        
        return vector

    def transform(self, X):
        super().transform()
        return self._vectorize(X)

    def fit_transform(self, X):
        super().fit_transform()
        return self._vectorize(X)
