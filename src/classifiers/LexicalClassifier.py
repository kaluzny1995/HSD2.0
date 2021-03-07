import numpy as np
import pandas as pd
import pickle

from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.notebook import tqdm

from ..classifiers.Classifier import Classifier
from ..dataframes.utils import combine_row_wisely
from ..utils.ext import load_ext_phrases
from ..extension.lemm import lemmatize_text
from ..analysis.poc import get_POC
from ..constants import (LABELS, LABELS_SMALL, SCORE_TYPES, POC_LABELS, OPTIM_POC_LABELS,
                       LC_MODEL_DIR, LC_CHART_DIR)


class LexicalClassifier(Classifier):
    def __init__(self, name='LexicalClassifier', k_folds=10):
        super(LexicalClassifier, self).__init__(name=name)
        self.k_folds = k_folds

        self._ext_phrases = load_ext_phrases()

        self._labels = LABELS
        self._labels_small = LABELS_SMALL
        self._score_types = SCORE_TYPES
        self._poc_labels = POC_LABELS

        self._skf_class = IterativeStratification

        self.best_f1s = dict(zip(self._labels_small, np.zeros(len(self._labels))))
        self.best_split_ids = dict(zip(self._labels_small, np.zeros(len(self._labels))))
        self.best_optimal_POCs = dict(zip(OPTIM_POC_LABELS, np.zeros(len(OPTIM_POC_LABELS))))

    def __str__(self):
        return self.name

    def _f_stats(self, df):
        thresholds = dict({})
        f_neg_scores = dict({})
        f_pos_scores = dict({})

        for label, lsmall in tqdm(zip(self._labels, self._labels_small), total=len(self._labels), leave=None):
            for sc in self._score_types:
                poc_scores = np.array(df[f'{lsmall}_POC_{sc}'])
                y_true = np.array(df[label])
                f0_scores, f1_scores = list([]), list([])

                u_poc_scores = np.unique(poc_scores)
                for poc_score in u_poc_scores:
                    y_pred = np.array([1 if ps >= poc_score else 0 for ps in poc_scores])

                    f_scores = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
                    f0_scores.append(f_scores[0])
                    f1_scores.append(f_scores[1])

                thresholds[f'{lsmall}_{sc}'] = u_poc_scores
                f_neg_scores[f'{lsmall}_{sc}'] = np.array(f0_scores)
                f_pos_scores[f'{lsmall}_{sc}'] = np.array(f1_scores)

        return thresholds, f_neg_scores, f_pos_scores

    def _optimal_thresholds(self, thrs, f_negs, f_poss):
        optims = dict({})

        for lsmall in self._labels_small:
            for sc in SCORE_TYPES:
                thr = thrs[f'{lsmall}_{sc}']
                f_neg = f_negs[f'{lsmall}_{sc}']
                f_pos = f_poss[f'{lsmall}_{sc}']

                optims[f'neg_{lsmall}_{sc}'] = thr[np.argmax(f_neg)]
                optims[f'pos_{lsmall}_{sc}'] = thr[np.argmax(f_pos)]
                del thr, f_neg, f_pos

        return optims

    def _test_data(self, df, optims):
        y_trues, y_preds = list([]), list([])

        for label, lsmall in zip(self._labels, self._labels_small):
            y_min = df[f'{lsmall}_POC_min'].values
            y_mean = df[f'{lsmall}_POC_mean'].values
            y_max = df[f'{lsmall}_POC_max'].values

            min_thr = optims[f'pos_{lsmall}_min']
            mean_thr = optims[f'pos_{lsmall}_mean']
            max_thr = optims[f'pos_{lsmall}_max']

            y_true = df[label].values
            y_pred = np.array([1 if mn >= min_thr and me >= mean_thr and mx >= max_thr else 0
                               for mn, me, mx in zip(y_min, y_mean, y_max)])
            y_trues.append(y_true)
            y_preds.append(y_pred)

            del y_min, y_mean, y_max, min_thr, mean_thr, max_thr

        return np.array(y_trues), np.array(y_preds)

    def fit(self, X, y):
        super().fit(X, y)
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.DataFrame:
            y = y.values

        skf = self._skf_class(n_splits=self.k_folds, order=1)
        for train_index, test_index in tqdm(skf.split(X, y), total=self.k_folds):
            df_train = combine_row_wisely([pd.DataFrame(X[train_index], columns=self._poc_labels),
                                           pd.DataFrame(y[train_index], columns=self._labels)])
            df_test = combine_row_wisely([pd.DataFrame(X[test_index], columns=self._poc_labels),
                                          pd.DataFrame(y[test_index], columns=self._labels)])

            thresholds, f_neg_scores, f_pos_scores = self._f_stats(df_train)
            optimal_POCs = self._optimal_thresholds(thresholds, f_neg_scores, f_pos_scores)
            trues, preds = self._test_data(df_test, optimal_POCs)

            for i, (lsmall, y_true, y_pred) in enumerate(zip(self._labels_small, trues, preds)):
                _, f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
                if self.best_f1s[lsmall] <= f1:
                    self.best_f1s[lsmall] = f1
                    self.best_split_ids[lsmall] = [train_index, test_index]
                    for sc in self._score_types:
                        for s in ['neg', 'pos']:
                            self.best_optimal_POCs[f'{s}_{lsmall}_{sc}'] = optimal_POCs[f'{s}_{lsmall}_{sc}']

    def _f_measure_lines(self, thrs, f0_s, f1_s, highlight_optims=True,
                         title='F measures for thresholds of min, mean and max POC.', save_file=None):
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))

        titles = ['Minimum POC', 'Mean POC', 'Maximum POC']
        for i in range(3):
            ax[i].set_title(titles[i])
            ax[i].set_ylabel('Score')
            ax[i].set_xlabel('Threshold')
            ax[i].plot(thrs[i], f0_s[i])
            ax[i].plot(thrs[i], f1_s[i])

            if highlight_optims:
                mx, my = thrs[i][np.argmax(f0_s[i])], np.max(f0_s[i])
                ax[i].scatter([mx], [my], s=200, marker='*')
                ax[i].annotate(f'({mx:.2f}, {my:.2f})', xy=(mx, my), fontsize=20)
                mx, my = thrs[i][np.argmax(f1_s[i])], np.max(f1_s[i])
                ax[i].scatter([mx], [my], s=200, marker='*')
                ax[i].annotate(f'({mx:.2f}, {my:.2f})', xy=(mx, my), fontsize=20)

        # custom legend
        legend_lines = [Line2D([0], [0], color='#e24a33', lw=2, label='F0'),
                        Line2D([0], [0], color='#348abd', lw=2, label='F1'),
                        Line2D([0], [0], marker='*', color='#e24a33', markersize=15, label='max F0'),
                        Line2D([0], [0], marker='*', color='#348abd', markersize=15, label='max F1')]
        fig.legend(handles=legend_lines[:2], loc='upper left')
        fig.legend(handles=legend_lines[2:], loc='upper right')

        fig.suptitle(title)
        if save_file:
            plt.savefig(save_file)
        plt.show()

    def plot_f_measure_lines(self, X, y):
        if not self._fit:
            raise ValueError(f'Classifier: {self.name} must be fit first!')
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.DataFrame:
            y = y.values

        for label, lsmall in zip(LABELS, LABELS_SMALL):
            ids = self.best_split_ids[lsmall][0]
            df = combine_row_wisely([pd.DataFrame(X[ids], columns=self._poc_labels),
                                     pd.DataFrame(y[ids], columns=self._labels)])
            thresholds, f_neg_scores, f_pos_scores = self._f_stats(df)

            self._f_measure_lines([thresholds[f'{lsmall}_{sc}'] for sc in SCORE_TYPES],
                                  [f_neg_scores[f'{lsmall}_{sc}'] for sc in SCORE_TYPES],
                                  [f_pos_scores[f'{lsmall}_{sc}'] for sc in SCORE_TYPES],
                                  title=f'F measures for thresholds of min, mean and max POC for class "{label}".',
                                  save_file=LC_CHART_DIR.replace('{}', f'f_measure_lines_{lsmall}'))

    def predict(self, X):
        super().predict(X)
        if type(X) == pd.DataFrame:
            X = X.values

        X = X.reshape(-1, len(self._labels_small), 3)
        pred = np.zeros((X.shape[0], len(self._labels_small)), dtype=np.uint8)

        for i in range(X.shape[0]):
            for j, lsmall in enumerate(self._labels_small):
                if X[i][j][0] >= self.best_optimal_POCs[f'pos_{lsmall}_min'] and \
                        X[i][j][1] >= self.best_optimal_POCs[f'pos_{lsmall}_mean'] and \
                        X[i][j][2] >= self.best_optimal_POCs[f'pos_{lsmall}_max']:
                    pred[i][j] = 1.

        return pred

    def test(self, text):
        super().test(text)
        lemm = lemmatize_text(text)
        poc = get_POC(lemm, self._ext_phrases)
        preds = self.predict(np.array(poc).reshape(1, -1))

        return preds

    def save(self, save_file=LC_MODEL_DIR):
        super().save(save_file)
        model_dict = dict({
            'f1_s': self.best_f1s,
            'split_ids': self.best_split_ids,
            'optims': self.best_optimal_POCs
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=LC_MODEL_DIR):
        super().load(load_file)
        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.best_f1s = model_dict['f1_s']
        self.best_split_ids = model_dict['split_ids']
        self.best_optimal_POCs = model_dict['optims']


