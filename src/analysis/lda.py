import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from tqdm.notebook import tqdm

from ..constants import LABELS_V_SMALL, LDA_MODEL_DIR, POLISH_STOPWORDS


def train_lda_models(phrs, n_topics=10):
    for label, phrases, in tqdm(zip(LABELS_V_SMALL, phrs), total=len(LABELS_V_SMALL)):

        cv = CountVectorizer(stop_words=POLISH_STOPWORDS)
        count_data = cv.fit_transform(phrases)

        lda_model = LDA(n_components=n_topics, n_jobs=-1)
        lda_model.fit(count_data)

        with open(LDA_MODEL_DIR.replace('{}', label), 'wb') as f:
            pickle.dump([lda_model, cv], f)


def lda_topics(lda_model, lda_cv, n_words=10):
    words = lda_cv.get_feature_names()

    topics = list([' '.join([words[i] for i in topic.argsort()[:-n_words - 1:-1]])
                   for topic in lda_model.components_])

    return topics
