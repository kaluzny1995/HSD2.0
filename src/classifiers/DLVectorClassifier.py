import numpy as np
import pandas as pd
import pickle

from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from ..classifiers.Classifier import Classifier
from ..extension.lemm import lemmatize_text
from ..nn.datasets import TweetsDataset
from ..constants import DLVC_MODEL_DIR, DLCV_MODEL_DIR


class DLVectorClassifier(Classifier):
    def __init__(self, name='DLVectorClassifier', short_name='DLVC', k_folds=10, weights=None,
                 vec_class=None, nn_class=None, nn_type='dense', vec_params=None, vec_output_dims=768,
                 vec_analysis=False, nn_hparams=None, nn_params=None, verbose=0):
        super(DLVectorClassifier, self).__init__(name=name)
        if not vec_class:
            raise ValueError('Class of the the vectorizer (vec_class) must be specified! Found None!')
        if not nn_class:
            raise ValueError('Class of the the neural network model (nn_class) must be specified! Found None!')
        if vec_analysis and nn_type != 'dense':
            raise ValueError('Vector analysis is applicable only for dense neural network type!')

        default_hyperparams = dict({
            '_train_bs': 60,
            '_valid_bs': 30,
            '_test_bs': 30,
            '_lr': 0.0005,
            '_epochs': 10,
        })

        self.k_folds = k_folds
        self.short_name = short_name
        self.weights = weights
        self.nn_type = nn_type

        self._skf_class = IterativeStratification
        self._vec_class = vec_class
        self._nn_class = nn_class
        self._vec_params = dict({}) if not vec_params else vec_params
        self._vec_output_dims = vec_output_dims
        self._nn_params = dict({}) if not nn_params else nn_params
        if not vec_analysis:
            self.default_save_file = DLVC_MODEL_DIR.replace('[]', self.nn_type).replace('{}', self.short_name)
            self.default_model_save_file = self.default_save_file.replace('.pkl', '.pt')
        else:
            self.default_save_file = DLCV_MODEL_DIR.replace('{}', self.short_name)
            self.default_model_save_file = self.default_save_file.replace('.pkl', '.pt')

        self._nn_hparams = default_hyperparams if not nn_hparams else nn_hparams
        self._train_bs = self._nn_hparams['_train_bs']
        self._valid_bs = self._nn_hparams['_valid_bs']
        self._test_bs = self._nn_hparams['_test_bs']
        self._lr = self._nn_hparams['_lr']
        self._epochs = self._nn_hparams['_epochs']

        self._model = None
        self.best_f = 0.
        self.best_split_ids = None
        self.metrics = None

        self._verbose = verbose

    def __str__(self):
        return self.name

    def _vectorize(self, X):
        vec = self._vec_class(**self._vec_params)
        vec.load()

        X = vec.transform(X)
        if len(X.shape) > 2:
            X = np.array([x.flatten() for x in X])

        return X

    def _adapt_shape(self, x):
        # determine nn input shape
        self._vec_output_dims = len(x[0])  # for dense nns
        self._nn_params['in_size'] = len(x[0])

        if any(list([self.nn_type.find(n) >= 0 for n in ['conv1d', 'recurrent', 'lstm', 'gru']])):
            x = np.array([xx[:self._vec_output_dims//3*3] for xx in x])  # truncate to number of dims divisible by 3
            x = x.reshape((-1, 3, self._vec_output_dims//3))

            if any(list([self.nn_type.find(n) >= 0 for n in ['recurrent', 'lstm', 'gru']])):  # for recurrent nns
                self._vec_output_dims = len(x[0][0])
                self._nn_params['in_size'] = len(x[0][0])
            else:  # for conv nns
                self._nn_params['input_dim'] = (3, len(x[0][0]))

        return x

    def _sigmoid(self, x):
        return np.array([1/(1 + np.exp(-xx)) for xx in x])

    def _calculate_metrics(self, pred, true):
        with torch.no_grad():
            p, t = pred.numpy(), true.numpy()
            p, t, = np.array(p, dtype=np.float64), np.array(t, dtype=np.float64)
            p = self._sigmoid(p).round()  # sigmoid to enable metrics calculation

            a = accuracy_score(y_pred=p, y_true=t)
            f0, f1 = f1_score(y_pred=p, y_true=t, labels=[0, 1], average=None, zero_division=1.)

        return a, f0, f1

    def _train(self, optim, train_dl, e):
        # training step
        self._model.train()
        ta, tf, tl = list([]), list([]), list([])

        t = tqdm(train_dl, leave=False)
        for step, (features, labels) in enumerate(t):
            optim.zero_grad()

            pred = self._model(features)
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(self.weights))(pred, labels)
            loss.backward()
            optim.step()

            a, f0, f1 = self._calculate_metrics(pred, labels)
            ta.append(a)
            tf.append((f0 + f1) / 2)
            tl.append(loss.item())
            t.set_postfix_str(f'Epoch: {e + 1}/{self._epochs} | Step: {step + 1}/{len(train_dl)} |' +
                              f' Train loss: {loss.item():.4} | Acc.: {a:.4} | F0/1: {f0:.4}/{f1:.4}')

        return np.mean(ta), np.mean(tf), np.mean(tl)

    def _eval(self, valid_dl, e):
        # validation step
        self._model.eval()
        va, vf, vl = list([]), list([]), list([])

        with torch.no_grad():
            t = tqdm(valid_dl, leave=False)
            for step, (features, labels) in enumerate(t):
                pred = self._model(features)
                loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(self.weights))(pred, labels)

                a, f0, f1 = self._calculate_metrics(pred, labels)
                va.append(a)
                vf.append((f0 + f1) / 2)
                vl.append(loss.item())
                t.set_postfix_str(f'Epoch: {e + 1}/{self._epochs} | Step: {step + 1}/{len(valid_dl)} |' +
                                  f' Valid. loss: {loss.item():.4} | Acc.: {a:.4} | F0/1: {f0:.4}/{f1:.4}')

        return np.mean(va), np.mean(vf), np.mean(vl)

    def _test(self, test_dl):
        # test step
        self._model.eval()

        with torch.no_grad():
            preds = list([])

            t = tqdm(test_dl, leave=False)
            for step, features in enumerate(t):
                pred = self._model(features)
                preds.append(pred)
            preds = torch.cat(preds, dim=0)

        return preds.numpy()

    def fit(self, X, y):
        super().fit(X, y)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()
        elif type(X) == np.ndarray:
            X = X.flatten()
        if type(y) == pd.DataFrame:
            y = y.values

        X = self._vectorize(X)
        X = self._adapt_shape(X)

        skf = self._skf_class(n_splits=self.k_folds, order=1)
        t = tqdm(skf.split(X, y), total=self.k_folds, leave=bool(self._verbose))
        best_fold_f = 0.
        for i, (train_index, valid_index) in enumerate(t):
            t.set_postfix_str(f'Fold: {i+1}/{self.k_folds}')
            train_ds = TweetsDataset(X[train_index], y[train_index])
            train_dl = DataLoader(train_ds, batch_size=self._train_bs, shuffle=False)
            valid_ds = TweetsDataset(X[valid_index], y[valid_index])
            valid_dl = DataLoader(valid_ds, batch_size=self._valid_bs, shuffle=False)

            self._model = self._nn_class(**self._nn_params)
            optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._lr)

            ta, tf, tl, va, vf, vl = list([]), list([]), list([]), list([]), list([]), list([])
            tt = tqdm(range(self._epochs), leave=False)
            for epoch in tt:
                tt.set_postfix_str(f'Epoch: {epoch+1}/{self._epochs}')
                a, f, ls = self._train(optim=optimizer, train_dl=train_dl, e=epoch)
                ta.append(a)
                tf.append(f)
                tl.append(ls)
                a, f, ls = self._eval(valid_dl=valid_dl, e=epoch)
                va.append(a)
                vf.append(f)
                vl.append(ls)

                if self.best_f <= f:
                    torch.save(self._model.state_dict(), self.default_model_save_file)
                    self.best_f = f
                    self.best_split_ids = [train_index, valid_index]

            if best_fold_f <= np.max(vf):
                self.metrics = np.array([ta, tf, tl, va, vf, vl])
                best_fold_f = np.max(vf)

        self._model.load_state_dict(torch.load(self.default_model_save_file))

    def plot_train_history_lines(self, title='Deep Learning model training history.', save_file=None):
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        data = self.metrics
        data_ids = [[0, 3], [1, 4], [2, 5]]
        labels = np.array([['train acc.', 'valid. acc.'], ['train F', 'valid. F'], ['train loss', 'valid. loss']])
        colors = np.array(['#f9766e', '#619dff', '#da72fb'])
        styles = np.array(['-', '--'])
        titles = ['Accuracy', 'Mean F measure', 'BCE loss']

        for i, (ids, ls) in enumerate(zip(data_ids, labels)):
            ax[i].plot(range(self._epochs), data[ids[0]], label=ls[0], color=colors[i], linestyle=styles[0])
            ax[i].plot(range(self._epochs), data[ids[1]], label=ls[1], color=colors[i], linestyle=styles[1])
            ax[i].legend(loc='best')

            ax[i].set_title(titles[i])
            ax[i].set_xticks(range(10))
            ax[i].set_xticklabels(range(1, 11))
            ax[i].set_xlabel('Epoch')
            if i != 2:
                ax[i].set_ylim([0., 1.])
            ax[i].set_ylabel('Score')

        plt.suptitle(title, fontsize=16)
        if save_file:
            plt.savefig(save_file)
        plt.show()

    def predict(self, X):
        super().predict(X)
        if type(X) == pd.DataFrame:
            X = X.values.flatten()
        elif type(X) == np.ndarray:
            X = X.flatten()

        X = self._vectorize(X)
        X = self._adapt_shape(X)

        test_ds = TweetsDataset(X, is_test=True)
        test_dl = DataLoader(test_ds, batch_size=self._test_bs, shuffle=False)

        y = self._test(test_dl=test_dl)
        y = self._sigmoid(y).round()

        return y

    def test(self, text):
        super().test(text)
        lemm = lemmatize_text(text)
        preds = self.predict(np.array([lemm]))

        return preds.flatten()

    def save(self, save_file=None, model_save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = self.default_save_file
        model_dict = dict({
            'f': self.best_f,
            'split_ids': self.best_split_ids,
            'metrics': self.metrics,
            'vec_out_dims': self._vec_output_dims,
            'in_size': self._nn_params['in_size'],
            'in_dim': None if not 'input_dim' in self._nn_params else self._nn_params['input_dim']
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

        if not model_save_file:
            model_save_file = self.default_model_save_file
        torch.save(self._model.state_dict(), model_save_file)

    def load(self, load_file=None, model_load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = self.default_save_file
        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.best_f = model_dict['f']
        self.best_split_ids = model_dict['split_ids']
        self.metrics = model_dict['metrics']
        self._vec_output_dims = model_dict['vec_out_dims']

        if not model_load_file:
            model_load_file = self.default_model_save_file
        self._nn_params['in_size'] = model_dict['in_size']
        if self.nn_type.find('conv1d') >= 0:
            self._nn_params['input_dim'] = model_dict['in_dim']  # adjust input dim for conv nn
        self._model = self._nn_class(**self._nn_params)
        self._model.load_state_dict(torch.load(model_load_file))
