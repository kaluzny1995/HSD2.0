from .classifiers.LexicalClassifier import LexicalClassifier
from .classifiers.SimpleMLClassifier import SimpleMLClassifier
from .classifiers.SimpleMLVectorClassifier import SimpleMLVectorClassifier
from .classifiers.DLVectorClassifier import DLVectorClassifier

import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier

from .vectorizers.TextOwnTrainedFTVectorizer import TextOwnTrainedFTVectorizer

from .nn.models import *


# best model: names, types, short names, params, classifier classes and kwargs
NAMES = ['Lexical Classifier',
         'Logistic Regression Feature Classifier', 'Stochastic Gradient Descent Vector Classifier',
         'Dense Neural Network Classifier', '1D Convolutional Neural Network Classifier', 'Simple Recurrent Neural Network Classifier', 'LSTM Neural Network Classifier', 'GRU Neural Network Classifier',
         '1D Convolutional Recurrent Neural Network Clasifier', '1D Convolutional LSTM Neural Network Clasifier', '1D Convolutional GRU Neural Network Clasifier',
         'Dense Neural Network Classifier', '1D Convolutional Neural Network Classifier',
         'LSTM Neural Network Classifier', 'GRU Neural Network Classifier',
         '1D Convolutional LSTM Neural Network Clasifier', '1D Convolutional GRU Neural Network Clasifier', ]
TYPES = ['lexical',
         'simple machine learning', 'simple machine learning',
         'simple deep learning', 'simple deep learning', 'simple deep learning', 'simple deep learning', 'simple deep learning',
         'complex deep learning', 'complex deep learning', 'complex deep learning',
         'advanced deep learning', 'advanced deep learning',
         'advanced deep learning', 'advanced deep learning',
         'advanced deep learning', 'advanced deep learning', ]
SHORT_NAMES = ['Lexical',
               'LRFC', 'SGDVC',
               'DNN', '1dCNN', 'RNN', 'LSTM', 'GRU',
               '1dCNN+RNN', '1dCNN+LSTM', '1dCNN+GRU',
               'DNN-HP', '1dCNN-HP',
               'LSTM-HP', 'GRU-HP',
               '1dCNN+LSTM-HP', '1dCNN+GRU-HP', ]
PARAMETERS = ['{}',
              '{penalty: l2, solver: liblinear, class_weight: balanced}', '{penalty: l2, class_weight: balanced}',
              '{hidden size: 300, dropout: 0.1, dense layers: 1}', '{channels: 32, kernel size: 3, conv. layers: 3}',
              '{rec. layers: 5, dropout: 0.1, bidirectional: False}', '{rec. layers: 1, dropout: 0.1, bidirectional: False}',
              '{rec. layers: 1, dropout: 0., bidirectional: True}',
              '{channels: 20, hidden size: 100, bidirectional: True}', '{channels: 32, hidden size: 100, bidirectional: True}',
              '{channels: 32, hidden size: 50, bidirectional: True}',
              '{_epochs: 50, _optim: torch.optim.AdamW, _optim_params: dict({lr: 0.01, amsgrad: False}), ' +
              '_sched: torch.optim.lr_scheduler.CyclicLR, _sched_params: dict({base_lr: 0.001, max_lr: 0.01, cycle_momentum: False}),}',
              '{_epochs: 50, _optim: torch.optim.AdamW, _optim_params: dict({amsgrad: False}), ' +
              '_sched: torch.optim.lr_scheduler.ReduceLROnPlateau, _sched_params: dict({patience: 5, factor: 0.97}),}',
              '{_epochs: 50, _optim: torch.optim.AdamW, _optim_params: dict({amsgrad: False}), ' +
              '_sched: torch.optim.lr_scheduler.ReduceLROnPlateau, _sched_params: dict({patience: 5, factor: 0.97}),}',
              '{_epochs: 50, _optim: torch.optim.AdamW, _optim_params: dict({lr: 0.01, amsgrad: False}), ' +
              '_sched: torch.optim.lr_scheduler.CyclicLR, _sched_params: dict({base_lr: 0.001, max_lr: 0.01, cycle_momentum: False}),}',
              '{_epochs: 50, _optim: torch.optim.AdamW, _optim_params: dict({amsgrad: True}), ' +
              '_sched: torch.optim.lr_scheduler.ReduceLROnPlateau, _sched_params: dict({patience: 5, factor: 0.97}),}',
              '{_epochs: 50, _optim: torch.optim.AdamW, _optim_params: dict({amsgrad: True}), ' +
              '_sched: torch.optim.lr_scheduler.ReduceLROnPlateau, _sched_params: dict({patience: 5, factor: 0.97}),}', ]

CLF_CLASSES = [LexicalClassifier,
               SimpleMLClassifier, SimpleMLVectorClassifier,
               DLVectorClassifier, DLVectorClassifier, DLVectorClassifier, DLVectorClassifier,
               DLVectorClassifier, DLVectorClassifier, DLVectorClassifier, DLVectorClassifier,
               DLVectorClassifier, DLVectorClassifier, DLVectorClassifier,
               DLVectorClassifier, DLVectorClassifier, DLVectorClassifier, ]
common_vkwargs = dict({'model_type': 's', 'short_name': 'super'})
CLF_KWARGS = [{'k_folds': 5},
              {'k_folds': 5, 'short_name': 'LRC-l2',
               'clf_class': LogisticRegression, **dict({'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced'})},
              {'k_folds': 5, 'short_name': 'SGD-l2',
               'vec_class': TextOwnTrainedFTVectorizer, 'clf_class': SGDClassifier,
               'vec_kwargs': {'length': 300, 'model_type': 's', 'short_name': 'super', 'verbose': 0},
               **dict({'penalty': 'l2', 'class_weight': 'balanced'})},

              {'short_name': '300-1-1', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': DenseNet,
               'nn_type': 'dense_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'hidden_size': 300, 'drop_coeff': 0.1, 'n_linear': 1})},
              {'short_name': '32-3-3', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': Conv1dNet,
               'nn_type': f'conv1d_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'out_channels': 32, 'kernel_size': 3, 'n_convs': 3})},
              {'short_name': '5-1-0', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': RecurrentNet,
               'nn_type': 'recurrent_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'n_layers': 5, 'drop_prob': 0.1, 'bidirectional': False})},
              {'short_name': '1-1-0', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': LSTMNet,
               'nn_type': 'lstm_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'n_layers': 1, 'drop_prob': 0.1, 'bidirectional': False})},
              {'short_name': '1-0-1', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': GRUNet,
               'nn_type': f'gru_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'n_layers': 1, 'drop_prob': 0., 'bidirectional': True})},
              {'short_name': '20-100-1', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': Conv1dRecurrentNet,
               'nn_type': 'conv1d_recurrent_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'nn_type': 'recurrent', 'out_channels': 20, 'hidden_size': 100, 'bidirectional': True})},
              {'short_name': '32-100-1', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': Conv1dRecurrentNet,
               'nn_type': 'conv1d_lstm_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'nn_type': 'lstm', 'out_channels': 32, 'hidden_size': 100, 'bidirectional': True})},
              {'short_name': '32-50-1', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': Conv1dRecurrentNet,
               'nn_type': 'conv1d_gru_w2', 'vec_params': common_vkwargs,
               'nn_params': dict({'nn_type': 'gru', 'out_channels': 32, 'hidden_size': 50, 'bidirectional': True})},

              {'short_name': f'dense_adamw-noams-cyc', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': DenseNet,
               'nn_type': 'hparams_dense_w2',
               'nn_hparams': dict({'_epochs': 50, '_optim': torch.optim.AdamW, '_optim_params': dict({'lr': 0.01, 'amsgrad': False}),
                                   '_sched': torch.optim.lr_scheduler.CyclicLR,
                                   '_sched_params': dict({'base_lr': 0.001, 'max_lr': 0.01, 'cycle_momentum': False}),}),
               'vec_params': common_vkwargs, 'nn_params': dict({'hidden_size': 500, 'drop_coeff': 0., 'n_linear': 1})},
              {'short_name': 'c1d_adamw-noams-rop', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': Conv1dNet,
               'nn_type': 'hparams_conv1d_w2',
               'nn_hparams': dict({'_epochs': 50, '_optim': torch.optim.AdamW, '_optim_params': dict({'amsgrad': False}),
                                   '_sched': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                   '_sched_params': dict({'patience': 5, 'factor': 0.97}),}),
               'vec_params': common_vkwargs, 'nn_params': dict({'out_channels': 32, 'kernel_size': 3, 'n_convs': 2})},
              {'short_name': 'lstm_adamw-noams-rop', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': LSTMNet,
               'nn_type': 'hparams_lstm_w2',
               'nn_hparams': dict({'_epochs': 50, '_optim': torch.optim.AdamW, '_optim_params': dict({'amsgrad': False}),
                                   '_sched': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                   '_sched_params': dict({'patience': 5, 'factor': 0.97}),}),
               'vec_params': common_vkwargs, 'nn_params': dict({'n_layers': 1, 'drop_prob': 0., 'bidirectional': True})},
              {'short_name': 'gru_adamw-noams-cyc', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': GRUNet,
               'nn_type': 'hparams_gru_w2',
               'nn_hparams': dict({'_epochs': 50, '_optim': torch.optim.AdamW, '_optim_params': dict({'lr': 0.01, 'amsgrad': False}),
                                   '_sched': torch.optim.lr_scheduler.CyclicLR,
                                   '_sched_params': dict({'base_lr': 0.001, 'max_lr': 0.01, 'cycle_momentum': False}),}),
               'vec_params': common_vkwargs, 'nn_params': dict({'n_layers': 1, 'drop_prob': 0., 'bidirectional': True})},
              {'short_name': '1dclstm_adamw-ams-rop', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': Conv1dRecurrentNet,
               'nn_type': 'hparams_conv1d_w2',
               'nn_hparams': dict({'_epochs': 50, '_optim': torch.optim.AdamW, '_optim_params': dict({'amsgrad': True}),
                                   '_sched': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                   '_sched_params': dict({'patience': 5, 'factor': 0.97}),}),
               'vec_params': common_vkwargs, 'nn_params': dict({'nn_type': 'lstm', 'out_channels': 8, 'hidden_size': 100, 'bidirectional': True})},
              {'short_name': '1dcgru_adamw-ams-rop', 'k_folds': 5, 'vec_class': TextOwnTrainedFTVectorizer, 'nn_class': Conv1dRecurrentNet,
               'nn_type': 'hparams_conv1d_w2',
               'nn_hparams': dict({'_epochs': 50, '_optim': torch.optim.AdamW, '_optim_params': dict({'amsgrad': True}),
                                   '_sched': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                   '_sched_params': dict({'patience': 5, 'factor': 0.97}),}),
               'vec_params': common_vkwargs, 'nn_params': dict({'nn_type': 'gru', 'out_channels': 8, 'hidden_size': 100, 'bidirectional': True})}, ]
