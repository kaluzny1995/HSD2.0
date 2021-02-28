import numpy as np
import pandas as pd

import pickle
from tqdm.notebook import tqdm

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from ..utils.ops import batch
from ..vectorizers.Vectorizer import Vectorizer
from ..constants import TOTBERTV_MODEL_DIR, BERT_OWN_MODEL_DIR, BERT_MODEL, ROBERTA_MODEL


# Based on solution: https://www.sbert.net/docs/training/overview.html
class TextOwnTrainedBERTVectorizer(Vectorizer):
    def __init__(self, name='TextOwnTrainedBERTVectorizer', batch_len=200, max_size=100, model_type='bert', short_name='BERT', verbose=0):
        super(TextOwnTrainedBERTVectorizer, self).__init__(name=name)
        assert model_type in ['bert', 'roberta'], 'Model type must be "bert" (BERT) or "roberta" (RoBERTa)!'

        self.type = model_type
        self.short_name = short_name
        self._verbose = verbose
        self._max_size = max_size  # size restriction due to insufficient memory of 16GB for full dataset
        self._epochs = 1

        self.device = 'cpu'
        self._batch_len = batch_len
        self._model = None

    def __str__(self):
        return self.name

    def fit(self, X):
        super().fit(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()

        name = BERT_MODEL if self.type == 'bert' else ROBERTA_MODEL
        self._model = SentenceTransformer(name, device=self.device)

        for bx in tqdm(batch(X, self._max_size), total=int(np.ceil(len(X)/self._max_size))):
            additionals = [InputExample(texts=bx, label=0.)]
            dataloader = DataLoader(additionals, shuffle=True, batch_size=4)
            loss = losses.CosineSimilarityLoss(self._model)

            self._model.fit(train_objectives=[(dataloader, loss)], epochs=self._epochs, warmup_steps=100)

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
            save_file = TOTBERTV_MODEL_DIR.replace('{}', self.type)

        self._model.save(BERT_OWN_MODEL_DIR.replace('{}', self.type))
        model_dict = dict({
            'name': self.name,
            'type': self.type
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = TOTBERTV_MODEL_DIR.replace('{}', self.type)

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
        self.type = model_dict['type']

        self._model = SentenceTransformer(BERT_OWN_MODEL_DIR.replace('{}', self.type), device=self.device)
