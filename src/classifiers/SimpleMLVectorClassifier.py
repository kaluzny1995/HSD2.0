import numpy as np
import pandas as pd
import pickle

from skmultilearn.model_selection import IterativeStratification
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm

from ..classifiers.Classifier import Classifier
from ..extension.lemm import lemmatize_text
from ..constants import SMLVC_MODEL_DIR, SMLCV_MODEL_DIR


class SimpleMLVectorClassifier(Classifier):
    def __init__(self, name='SimpleMLVectorClassifier', short_name='SMLVC', k_folds=10, vec_class=None, clf_class=None, vec_analysis=False, verbose=0, vec_kwargs=dict({}), **clf_kwargs):
        super(SimpleMLVectorClassifier, self).__init__(name=name)
        if not vec_class:
            raise ValueError('Class of the the vectorizer (vec_class) must be specified! Found None!')
        if not clf_class:
            raise ValueError('Class of the the classifier (clf_class) must be specified! Found None!')

        self.k_folds = k_folds
        self.short_name = short_name

        self._skf_class = IterativeStratification
        self._vec_class = vec_class
        self._clf_class = clf_class
        self._vec_kwargs = vec_kwargs
        self._clf_kwargs = clf_kwargs
        if not vec_analysis:
            self.default_save_file = SMLVC_MODEL_DIR.replace('{}', self.short_name)
        else:
            self.default_save_file = SMLCV_MODEL_DIR.replace('{}', self.short_name)

        self.best_f = 0.
        self.best_clf = None
        self.best_split_ids = None

        self._verbose = verbose

    def __str__(self):
        return self.name

    def _vectorize(self, X):
        vec = self._vec_class(**self._vec_kwargs)
        vec.load()

        X = vec.transform(X)
        if len(X.shape) > 2:
            X = np.array([x.flatten() for x in X])

        return X

    def fit(self, X, y):
        super().fit(X, y)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()
        elif type(X) == np.ndarray:
            X = X.flatten()
        if type(y) == pd.DataFrame:
            y = y.values

        X = self._vectorize(X)

        skf = self._skf_class(n_splits=self.k_folds, order=1)
        leave = None if not self._verbose else True
        for train_index, test_index in tqdm(skf.split(X, y), total=self.k_folds, leave=leave):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #if self.short_name.lower().split('-')[0] in ['sgd', 'lrc']:
            if self._clf_class in [SGDClassifier, LogisticRegression]:
                clf = OneVsRestClassifier(self._clf_class(**self._clf_kwargs))
            else:
                clf = self._clf_class(**self._clf_kwargs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            f0, f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
            f = (f0 + f1) / 2
            if self.best_f <= f:
                self.best_f = f
                self.best_clf = clf
                self.best_split_ids = [train_index, test_index]

    def predict(self, X):
        super().predict(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()
        elif type(X) == np.ndarray:
            X = X.flatten()

        X = self._vectorize(X)

        return self.best_clf.predict(X)

    def test(self, text):
        super().test(text)
        lemm = lemmatize_text(text)
        preds = self.predict(np.array([lemm]))

        return preds

    def save(self, save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = self.default_save_file

        model_dict = dict({
            'name': self.name,
            'short_name': self.short_name,
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
        self.name = model_dict['name']
        self.short_name = model_dict['short_name']
        self.best_f = model_dict['f']
        self.best_split_ids = model_dict['split_ids']
        self.best_clf = model_dict['clf']
