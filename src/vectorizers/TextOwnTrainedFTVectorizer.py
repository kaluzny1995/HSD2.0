import numpy as np
import pandas as pd

import pickle

import fasttext

from ..vectorizers.Vectorizer import Vectorizer
from ..constants import TOTFTV_MODEL_DIR, FT_OWN_MODEL_DIR, FT_DATA_DIR


class TextOwnTrainedFTVectorizer(Vectorizer):
    def __init__(self, name='TextOwnTrainedFTVectorizer', length=300, epochs=10, model_type='u', short_name='unsuper', verbose=0):
        super(TextOwnTrainedFTVectorizer, self).__init__(name=name)
        assert model_type in ['u', 's'], 'Model type must be "u" (unsupervisedly trained) or "s" (supervisedly trained)!'

        self.length = length
        self._epochs = epochs
        self.type = model_type
        self.short_name = short_name
        self._verbose = verbose

        self._model = None

    def __str__(self):
        return self.name

    def _prepare_data(self, X):
        with open(FT_DATA_DIR.replace('{}', self.type), 'w') as f:
            for x in X:
                if self.type == 'u':
                    f.write(x[0] + '\n')
                else:
                    label_string = ''.join([f'__label__{i+1} ' for i, l in enumerate(x[1:]) if int(l) == 1])
                    if not label_string:
                        label_string = '__label__0 '
                    f.write(label_string + x[0] + '\n')

    def fit(self, X):
        super().fit(X)
        if type(X) == pd.DataFrame:
            X = X.values

        self._prepare_data(X)
        file = FT_DATA_DIR.replace('{}', self.type)
        if self._verbose:
            print(f'Training model {"unsupervisedly" if self.type == "u" else "supervisedly"}...')
        if self.type == 'u':
            self._model = fasttext.train_unsupervised(file, dim=self.length, minn=2, maxn=10,
                                                      epoch=self._epochs, lr=0.5)
        else:
            self._model = fasttext.train_supervised(file, dim=self.length, wordNgrams=2, bucket=200000,
                                                    epoch=self._epochs, lr=0.5, loss='ova')
        if self._verbose:
            print('Training finished.')

    def transform(self, X):
        super().transform(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()

        return np.array([self._model.get_sentence_vector(x) for x in X])

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)

        if type(X) == pd.DataFrame:
            X = X.values
        X = np.array([x[0] for x in X])

        return self.transform(X)

    def save(self, save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = TOTFTV_MODEL_DIR.replace('{}', self.type)

        self._model.save_model(FT_OWN_MODEL_DIR.replace('{}', self.type))
        model_dict = dict({
            'name': self.name,
            'type': self.type
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = TOTFTV_MODEL_DIR.replace('{}', self.type)

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
        self.type = model_dict['type']

        self._model = fasttext.load_model(FT_OWN_MODEL_DIR.replace('{}', self.type))
