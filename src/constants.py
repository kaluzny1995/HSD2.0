# lists
with open('./data/other/polish_stopwords.txt', 'r') as f:\
    POLISH_STOPWORDS = f.read().split('\n')[:-1]
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
WEEKDAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

LABELS = ['wyzywanie', 'grożenie', 'wykluczanie', 'odczłowieczanie', 'poniżanie', 'stygmatyzacja', 'szantaż']
LABELS_SMALL = ['wyz', 'groz', 'wyk', 'odcz', 'pon', 'styg', 'szan']
LABELS_V = LABELS + ['wulgaryzm']
LABELS_V_SMALL = LABELS_SMALL + ['vulg']

POPULARITY_LABELS = ['replies_count', 'retweets_count', 'likes_count']

ANNOTATED_COLS = ['id', 'wydźwięk', 'klucze'] + LABELS
INFOS_COLS = ['new_id', 'date', 'time', 'user_id', 'username', 'name', 'tweet', 'emojis', 'emoticons',
              'mentions', 'hashtags', 'reply_to'] + POPULARITY_LABELS
COMBINED_COLS = ANNOTATED_COLS[:1] + INFOS_COLS[1:] + ANNOTATED_COLS[1:]

SCORE_TYPES = ['min', 'mean', 'max']
POC_LABELS = [f'{lsmall}_POC_{sc}' for sc in SCORE_TYPES for lsmall in LABELS_SMALL]
OPTIM_POC_LABELS = [f'{s}_{lsmall}_{sc}' for s in ['neg', 'pos'] for sc in SCORE_TYPES for lsmall in LABELS_SMALL]


# paths - phrases
HATEFUL_RAW_DIR = 'data/hateful/raw_{}.txt'
HATEFUL_LEMM_DIR = 'data/hateful/lemm_{}.txt'
HATEFUL_EXT_DIR = 'data/hateful/ext_{}.txt'
VULGARS_RAW_DIR = 'data/vulgars/raw_{}.txt'
VULGARS_LEMM_DIR = 'data/vulgars/lemm_{}.txt'
VULGARS_EXT_DIR = 'data/vulgars/ext_{}.txt'

# paths - tweets
RAW_PATH = 'data/tweets_sady/main/sady_infos_raw.csv'
ANNOTATED_PATH = 'data/tweets_sady/main/sady_date_annotated.csv'
SANITIZED_PATH = 'data/tweets_sady/main/sady_infos_sanitized.csv'
COMBINED_PATH = 'data/tweets_sady/main/sady_combined.csv'

# paths - supplement tweets
SUPPLEMENT_RAW_DIR = 'data/tweets_supplement/main/sady_{}_raw.csv'
SUPPLEMENT_SANITIZED_DIR = 'data/tweets_supplement/main/sady_{}_sanitized.csv'

# paths - processed tweets
LEMMAS_PATH = 'data/tweets_sady/processed/lemmas.csv'
DUPLICATED_PATH = 'data/tweets_sady/processed/sady_duplicated.csv'
POC_SCORES_PATH = 'data/tweets_sady/processed/poc_scores.csv'
TOPIC_POC_SCORES_PATH = 'data/tweets_sady/processed/topic_poc_scores.csv'
OTHER_SCORES_PATH = 'data/tweets_sady/processed/other_scores.csv'

# paths - models
LDA_MODEL_DIR = 'models/lda/lda_{}.pkl'
LC_MODEL_DIR = 'models/classification/lexical/lex.pkl'
SMLC_MODEL_DIR = 'models/classification/simple_ml/sml_{}.pkl'
CHV_MODEL_DIR = 'models/vectorization/char/chv.pkl'
WSBV_MODEL_DIR = 'models/vectorization/word/wsbv.pkl'
WPTV_MODEL_DIR = 'models/vectorization/word/wptv_{}.pkl'
WOTV_MODEL_DIR = 'models/vectorization/word/wotv_{}.pkl'
TTFIDF_MODEL_DIR = 'models/vectorization/text/ttfidf_{}.pkl'
TPTFTV_MODEL_DIR = 'models/vectorization/text/tptftv.pkl'
TOTFTV_MODEL_DIR = 'models/vectorization/text/totftv_{}.pkl'
TPTBERTV_MODEL_DIR = 'models/vectorization/text/tptbertv_{}.pkl'
TOTBERTV_MODEL_DIR = 'models/vectorization/text/totbertv_{}.pkl'
SMLVC_MODEL_DIR = 'models/classification/simple_vec_ml/smlv_ft_{}.pkl'
SMLCV_MODEL_DIR = 'models/classification/simple_ml_vecs/smlc_rfc_{}.pkl'
DLVC_MODEL_DIR = 'models/classification/vec_dl/[]/dlvc_{}.pkl'
DLCV_MODEL_DIR = 'models/classification/dl_vecs/dlv_{}_dense.pkl'

# paths - charts
IDA_CHART_DIR = 'charts/01. initial_data_analysis/{}.png'
SPA_CHART_DIR = 'charts/02. statistical_primary_analysis/{}.png'
EDA_CHART_DIR = 'charts/03. extended_data_analysis/{}.png'
LC_CHART_DIR = 'charts/04. lexical_classifier/{}.png'
SMLC_CHART_DIR = 'charts/05. simple_ml_classifier/{}.png'
SMLVC_CHART_DIR = 'charts/06. simple_ml_vector_classifier/{}.png'
SMLCV_CHART_DIR = 'charts/07. simple_ml_classifier_vectors/{}.png'
DLCV_CHART_DIR = 'charts/08. dl_classifier_vectors/{}.png'
DLDC_CHART_DIR = 'charts/09a. dl_dense_classifier/{}.png'
DLCC_CHART_DIR = 'charts/09b. dl_conv1d_classifier/{}.png'
DLRC_CHART_DIR = 'charts/09c. dl_recurrent_classifier/{}.png'
DLLSTMC_CHART_DIR = 'charts/09d. dl_lstm_classifier/{}.png'
DLGRUC_CHART_DIR = 'charts/09e. dl_gru_classifier/{}.png'

# paths - other
WORDNET_PATH = 'models/plwordnet_3_0/plwordnet-3.0.xml'
W2V_MODEL_DIR = 'models/word2vec/nkjp-lemmas-all-100-{}-hs.txt.gz'
W2V_OWN_MODEL_DIR = 'models/word2vec/w2v_{}.bin'
FT_MODEL_DIR = 'models/fasttext/kgr10_orths.vec.bin'
FT_DATA_DIR = 'models/fasttext/ft_data_{}.txt'
FT_OWN_MODEL_DIR = 'models/fasttext/ft_{}.bin'
BERT_OWN_MODEL_DIR = 'models/bert/bert_{}'

# other
TAGGER_MODEL = 'polish-herbert-base'
SPACY_PL_MODEL = 'pl_model'
HYPHENATION_MODEL = 'pl_PL'
BERT_MODEL = 'nli-bert-base'
ROBERTA_MODEL = 'nli-roberta-base'
