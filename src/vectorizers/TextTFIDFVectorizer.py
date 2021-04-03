import numpy as np
import pandas as pd

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from ..vectorizers.Vectorizer import Vectorizer
from ..constants import TTFIDF_MODEL_DIR, POLISH_STOPWORDS


class TextTFIDFVectorizer(Vectorizer):
    def __init__(self, name='TextTFIDFVectorizer', length=100, model_type='tfidf', short_name='TFIDF'):
        super(TextTFIDFVectorizer, self).__init__(name=name)
        assert model_type in ['tf', 'tfidf'], 'Model type must be "tf" (Term Frequency) or "tfidf" (Term Frequency with Inversed Document Frequency)!'

        self.length = length
        self.type = model_type
        self.short_name = short_name

        self._model = None

    def __str__(self):
        return self.name

    def fit(self, X):
        super().fit(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()

        use_idf = self.type == 'tfidf'
        self._model = TfidfVectorizer(max_features=self.length, ngram_range=(1, 10), analyzer='char', use_idf=use_idf,
                                      stop_words=POLISH_STOPWORDS)
        self._model.fit(X)

    def transform(self, X):
        super().transform(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()

        sparse_vectors = self._model.transform(X)
        return np.array([sv.toarray().flatten() for sv in sparse_vectors])

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)
        return self.transform(X)

    def save(self, save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = TTFIDF_MODEL_DIR.replace('{}', self.type)

        model_dict = dict({
            'name': self.name,
            'vec': self._model
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = TTFIDF_MODEL_DIR.replace('{}', self.type)

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
        self._model = model_dict['vec']
