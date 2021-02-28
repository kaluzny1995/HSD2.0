import numpy as np
import pandas as pd
import warnings
import multiprocessing
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phrases, Phraser

import pickle

from ..vectorizers.Vectorizer import Vectorizer
from ..constants import WOTV_MODEL_DIR, W2V_OWN_MODEL_DIR, LEMMAS_PATH


# Based on solution: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
class WordOwnTrainedVectorizer(Vectorizer):
    def __init__(self, name='WordOwnTrainedVectorizer', length=100, model_type='cbow', short_name='CBoW', verbose=0):
        super(WordOwnTrainedVectorizer, self).__init__(name=name)
        assert model_type in ['cbow', 'skipg'], 'Model type must be "cbow" (CBoW) or "skipg" (SkipGram)!'

        self.short_name = short_name
        self.type = model_type
        self._model = None
        self.length = length
        self.wv_length = 100
        self._verbose = verbose

        self.frequencies = dict({})

    def __str__(self):
        return self.name

    def fit(self, X):
        super().fit(X)
        if type(X) == pd.DataFrame:
            X = X.values

        phrases = Phrases(X, min_count=30, progress_per=10000)
        bigram = Phraser(phrases)
        sentences = bigram[X]

        for sent in sentences:
            for i in sent:
                if i not in self.frequencies:
                    self.frequencies[i] = 0
                self.frequencies[i] += 1

        sg = 1 if self.type == 'skipg' else 0
        cores = multiprocessing.cpu_count()
        self._model = Word2Vec(min_count=20,
                               window=2,
                               size=self.wv_length,
                               sample=6e-5,
                               alpha=0.03,
                               min_alpha=0.0007,
                               negative=20,
                               sg=sg,
                               workers=cores-1)

        if self._verbose:
            print('Building vocab...')
        self._model.build_vocab(sentences, progress_per=10000)
        if self._verbose:
            print('Training model...')
        self._model.train(sentences, total_examples=self._model.corpus_count, epochs=30, report_delay=1)
        if self._verbose:
            print('Training finished.')
        self._model.init_sims(replace=True)

    def transform(self, X):
        super().transform(X)
        if type(X) == pd.DataFrame:
            X = X.values

        vectors = list([])
        for x in X:
            text_vector = np.array([np.zeros(self.wv_length) if word not in self._model.wv.vocab else self._model.wv[word] for word in x.split(' ')])
            if len(text_vector) > self.length:
                warnings.warn(f'Desired length of vector greater than vector length. Excessive text will be truncated!',
                              category=RuntimeWarning, stacklevel=1)
                text_vector = text_vector[:self.length]
                #vectors.append(text_vector[:self.length])
            elif len(text_vector) < self.length:
                text_vector = np.concatenate([text_vector, np.zeros((self.length-len(text_vector), self.wv_length))], axis=0)
                #vectors.append(np.concatenate([text_vector, np.zeros((self.length-len(text_vector), self.wv_length))], axis=0))
            #else:
                #vectors.append(text_vector)
            vectors.append(text_vector.flatten())
        vectors = np.array(vectors)

        return vectors

    def fit_transform(self, X):
        super().fit_transform(X)
        self.fit(X)
        return self.transform(X)

    def save(self, save_file=None):
        super().save(save_file)
        if not save_file:
            save_file = WOTV_MODEL_DIR.replace('{}', self.short_name)

        self._model.wv.save_word2vec_format(W2V_OWN_MODEL_DIR.replace('{}', self.short_name), binary=True)
        model_dict = dict({
            'name': self.name,
            'type': self.type,
            'freq': self.frequencies
        })
        with open(save_file, 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self, load_file=None):
        super().load(load_file)
        if not load_file:
            load_file = WOTV_MODEL_DIR.replace('{}', self.short_name)

        with open(load_file, 'rb') as f:
            model_dict = pickle.load(f)
        self.name = model_dict['name']
        self.type = model_dict['type']
        self.frequencies = model_dict['freq']

        self._model = KeyedVectors.load_word2vec_format(W2V_OWN_MODEL_DIR.replace('{}', self.short_name), binary=True)
