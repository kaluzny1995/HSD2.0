import numpy as np
import pandas as pd

import pickle
from tqdm.notebook import tqdm

from sentence_transformers import SentenceTransformer

from ..utils.ops import batch
from ..vectorizers.Vectorizer import Vectorizer
from ..constants import TPTBERTV_MODEL_DIR, BERT_MODEL, ROBERTA_MODEL


class TextPretrainedBERTVectorizer(Vectorizer):
    def __init__(self, name='TextPretrainedBERTVectorizer', batch_len=200, model_type='bert', short_name='BERT', verbose=0):
        super(TextPretrainedBERTVectorizer, self).__init__(name=name)
        assert model_type in ['bert', 'roberta'], 'Model type must be "bert" (BERT) or "roberta" (RoBERTa)!'

        self.type = model_type
        self.short_name = short_name
        self._verbose = verbose

        self.device = 'cpu'
        self._batch_len = batch_len
        self._model = None

    def __str__(self):
        return self.name

    def fit(self, X):
        super().fit(X)
        name = BERT_MODEL if self.type == 'bert' else ROBERTA_MODEL
        self._model = SentenceTransformer(name, device=self.device)

    def transform(self, X):
        super().transform(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()

        leave = None if not self._verbose else True
        return np.concatenate([self._model.encode(bx) for bx in tqdm(batch(X, self._batch_len),
                                                                     total=int(np.ceil(len(X)/self._batch_len)),
                                                                     leave=leave)],
                              axis=0)

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)
        return self.transform(X)

    def save(self, save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = TPTBERTV_MODEL_DIR.replace('{}', self.type)

        model_dict = dict({
            'name': self.name,
            'type': self.type
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = TPTBERTV_MODEL_DIR.replace('{}', self.type)

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
        self.type = model_dict['type']

        name = BERT_MODEL if self.type == 'bert' else ROBERTA_MODEL
        self._model = SentenceTransformer(name, device=self.device)
