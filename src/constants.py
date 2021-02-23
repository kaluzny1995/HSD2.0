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

ANNOTATED_COLS = ['id', 'wydźwięk', 'klucze'] + LABELS
INFOS_COLS = ['new_id', 'date', 'time', 'user_id', 'username', 'name', 'tweet', 'emojis', 'emoticons',
              'mentions', 'hashtags', 'reply_to', 'replies_count', 'retweets_count', 'likes_count']
COMBINED_COLS = ANNOTATED_COLS[:1] + INFOS_COLS[1:] + ANNOTATED_COLS[1:]

SCORE_TYPES = ['min', 'mean', 'max']
POC_LABELS = [f'{lsmall}_POC_{sc}' for sc in SCORE_TYPES for lsmall in LABELS_SMALL]
OPTIM_POC_LABELS = [f'{s}_{lsmall}_{sc}' for s in ['neg', 'pos'] for sc in SCORE_TYPES for lsmall in LABELS_SMALL]


# paths
HATEFUL_RAW_DIR = 'data/hateful/raw_{}.txt'
HATEFUL_LEMM_DIR = 'data/hateful/lemm_{}.txt'
HATEFUL_EXT_DIR = 'data/hateful/ext_{}.txt'
VULGARS_RAW_DIR = 'data/vulgars/raw_{}.txt'
VULGARS_LEMM_DIR = 'data/vulgars/lemm_{}.txt'
VULGARS_EXT_DIR = 'data/vulgars/ext_{}.txt'

LDA_MODEL_DIR = 'models/lda/lda_{}.pkl'
LC_MODEL_DIR = 'models/lexical/lex.pkl'
SMLC_MODEL_DIR = 'models/simple_ml/sml_{}.pkl'

RAW_PATH = 'data/tweets_sady/main/sady_infos_raw.csv'
ANNOTATED_PATH = 'data/tweets_sady/main/sady_date_annotated.csv'
SANITIZED_PATH = 'data/tweets_sady/main/sady_infos_sanitized.csv'
COMBINED_PATH = 'data/tweets_sady/main/sady_combined.csv'

SUPPLEMENT_RAW_DIR = 'data/tweets_supplement/main/sady_{}_raw.csv'
SUPPLEMENT_SANITIZED_DIR = 'data/tweets_supplement/main/sady_{}_sanitized.csv'

LEMMAS_PATH = 'data/tweets_sady/processed/lemmas.csv'
DUPLICATED_PATH = 'data/tweets_sady/processed/sady_duplicated.csv'
POC_SCORES_PATH = 'data/tweets_sady/processed/poc_scores.csv'
TOPIC_POC_SCORES_PATH = 'data/tweets_sady/processed/topic_poc_scores.csv'
OTHER_SCORES_PATH = 'data/tweets_sady/processed/other_scores.csv'

WORDNET_PATH = 'models/plwordnet_3_0/plwordnet-3.0.xml'

# other
TAGGER_MODEL = 'polish-herbert-base'
SPACY_PL_MODEL = 'pl_model'
HYPHENATION_MODEL = 'pl_PL'
