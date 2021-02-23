import numpy as np
import pandas as pd
import pickle

from skmultilearn.model_selection import IterativeStratification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm

from ..classifiers.Classifier import Classifier
from ..utils.ext import load_ext_phrases
from ..extension.lemm import lemmatize_text
from ..analysis.poc import get_POC
from ..analysis.topic_poc import get_topic_POC
from ..analysis.other import get_other
from ..constants import SMLC_MODEL_DIR


class SimpleMLClassifier(Classifier):
    def __init__(self, name='SimpleMLClassifier', short_name='SMLC', k_folds=10, clf_class=None, verbose=1, **clf_kwargs):
        super(SimpleMLClassifier, self).__init__(name=name)
        if not clf_class:
            raise ValueError('Class of the the classifier (clf_class) must be specified! Found None!')

        self.k_folds = k_folds
        self.short_name = short_name

        self._ext_phrases = load_ext_phrases(load_vulg=True)

        self._skf_class = IterativeStratification
        self._clf_class = clf_class
        self._clf_kwargs = clf_kwargs
        self.default_save_file = SMLC_MODEL_DIR.replace('{}', self.short_name)

        self.best_f = 0.
        self.best_clf = None
        self.best_split_ids = None

        self._verbose = verbose

    def __str__(self):
        return self.name

    def fit(self, X, y):
        super().fit(X, y)
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.DataFrame:
            y = y.values

        skf = self._skf_class(n_splits=self.k_folds, order=1)
        leave = None if not self._verbose else True
        for train_index, test_index in tqdm(skf.split(X, y), total=self.k_folds, leave=leave):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if self.short_name.lower().split('-')[0] in ['sgd', 'lrc']:
                clf = OneVsRestClassifier(self._clf_class(**self._clf_kwargs))
            else:
                clf = self._clf_class(**self._clf_kwargs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            f0, f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
            f = (f0 + f1)/2
            if self.best_f <= f:
                self.best_f = f
                self.best_clf = clf
                self.best_split_ids = [train_index, test_index]

    def predict(self, X):
        super().predict(X)
        return self.best_clf.predict(X)

    def test(self, text):
        super().test(text)
        lemm = lemmatize_text(text)
        poc = get_POC(lemm, self._ext_phrases)
        topic_poc = get_topic_POC(lemm, n_words=20)
        other = get_other(text, lemm)

        combined = np.concatenate([poc, topic_poc, other]).reshape(1, -1)
        preds = self.predict(combined)

        return preds

    def save(self, save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = self.default_save_file

        model_dict = dict({
            'f': self.best_f,
            'split_ids': self.best_split_ids,
            'clf': self.best_clf
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = self.default_save_file

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.best_f = model_dict['f']
        self.best_split_ids = model_dict['split_ids']
        self.best_clf = model_dict['clf']
