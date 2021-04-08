import numpy as np
import pandas as pd

import pickle

import fasttext

from ..vectorizers.Vectorizer import Vectorizer
from ..constants import FT_MODEL_DIR, TPTFTV_MODEL_DIR


class TextPretrainedFTVectorizer(Vectorizer):
    def __init__(self, name='TextPretrainedFTVectorizer', length=300, short_name='tptftv'):
        super(TextPretrainedFTVectorizer, self).__init__(name=name)
        self.length = length
        self.short_name = short_name

        self._model = None

    def __str__(self):
        return self.name

    def fit(self, X):
        super().fit(X)
        self._model = fasttext.load_model(FT_MODEL_DIR)

    def transform(self, X):
        super().transform(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()

        return np.array([self._model.get_sentence_vector(str(x).replace('\n', '')) for x in X])

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)
        return self.transform(X)

    def save(self, save_file=TPTFTV_MODEL_DIR):
        super().save(save_file)

        model_dict = dict({
            'name': self.name,
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=TPTFTV_MODEL_DIR):
        super().load(load_file)

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
        self._model = fasttext.load_model(FT_MODEL_DIR)
