import numpy as np
import warnings
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

import pickle

from ..vectorizers.Vectorizer import Vectorizer
from ..constants import WPTV_MODEL_DIR, W2V_MODEL_DIR


class WordPretrainedVectorizer(Vectorizer):
    def __init__(self, name='WordPretrainedVectorizer', length=100, model_type='cbow', short_name='CBoW'):
        super(WordPretrainedVectorizer, self).__init__(name=name)
        assert model_type in ['cbow', 'skipg'], 'Model type must be "cbow" (CBoW) or "skipg" (SkipGram)!'

        self.short_name = short_name
        self.type = model_type
        self._model = None
        self._pca = None
        self.length = length
        self.wv_length = 100
        self.pca_length = 100

    def __str__(self):
        return self.name

    def _vectorize(self, X):
        vectors = list([])
        for x in X:
            text_vector = np.array(
                [np.zeros(self.wv_length) if word not in self._model.vocab else self._model[word] for word in
                 x.split(' ')])
            if len(text_vector) > self.length:
                warnings.warn(f'Desired length of vector greater than vector length. Excessive text will be truncated!',
                              category=RuntimeWarning, stacklevel=1)
                text_vector = text_vector[:self.length]
            elif len(text_vector) < self.length:
                text_vector = np.concatenate([text_vector, np.zeros((self.length - len(text_vector), self.wv_length))],
                                             axis=0)
            vectors.append(text_vector.flatten())

        return np.array(vectors)

    def fit(self, X):
        super().fit(X)
        self._model = KeyedVectors.load_word2vec_format(W2V_MODEL_DIR.replace('{}', self.type), binary=False)
        self._pca = PCA(n_components=self.pca_length)

        vectors = self._vectorize(X)
        self._pca.fit(vectors)

    def transform(self, X):
        super().transform(X)
        vectors = self._vectorize(X)

        vectors = self._pca.transform(vectors)

        return vectors

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)
        return self.transform(X)

    def save(self, save_file=None, pca_save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = WPTV_MODEL_DIR.replace('{}', self.short_name)
        if not pca_save_file:
            pca_save_file = WPTV_MODEL_DIR.replace('{}', self.short_name + '-pca')

        model_dict = dict({
            'name': self.name,
            'type': self.type
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)
        with open(pca_save_file, 'wb') as f:
            pickle.dump(self._pca, f)

    def load(self, load_file=None, pca_load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = WPTV_MODEL_DIR.replace('{}', self.short_name)
        if not pca_load_file:
            pca_load_file = WPTV_MODEL_DIR.replace('{}', self.short_name + '-pca')

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        with open(pca_load_file, 'rb') as f:
            self._pca = pickle.load(f)
        self.name = model_dict['name']
        self.type = model_dict['type']

        self._model = KeyedVectors.load_word2vec_format(W2V_MODEL_DIR.replace('{}', self.type), binary=False)
