# Project directory structure
**WARNING: Not all files or directories are in repository due to their size!**

```bash
├── 00. ReadMe.ipynb
├── 01. WebScraping.ipynb
├── 02. TweetSanitizer.ipynb
├── 03. InitialDataAnalysis.ipynb
├── 04. StatisticalPrimaryDataAnalysis.ipynb
├── 05. DataDuplicator.ipynb
├── 06. HatefulAndVulgarPhrasesExtension.ipynb
├── 07. ExtendedDataAnalysis.ipynb
├── 08. AdvancedDataAnalysis.ipynb
├── 09. LexicalClassifier.ipynb
├── 10. SimpleMLClassifier.ipynb
├── charts
│   ├── advanced_analysis
│   ├── initial_data_analysis
│   ├── lexical_classifier
│   ├── schemes
│   │   ├── HSD2.0_scheme01.png
│   │   ├── HSD2.0_scheme02.png
│   │   └── HSD2.0_scheme03.png
│   ├── simple_ml_classifier
│   └── statistical_primary_analysis
├── data
│   ├── hateful
│   │   ├── ext_groz.txt
│   │   ├── ext_odcz.txt
│   │   ├── ext_pon.txt
│   │   ├── ext_styg.txt
│   │   ├── ext_szan.txt
│   │   ├── ext_wyk.txt
│   │   ├── ext_wyz.txt
│   │   ├── lemm_groz.txt
│   │   ├── lemm_odcz.txt
│   │   ├── lemm_pon.txt
│   │   ├── lemm_styg.txt
│   │   ├── lemm_szan.txt
│   │   ├── lemm_wyk.txt
│   │   ├── lemm_wyz.txt
│   │   ├── raw_groz.txt
│   │   ├── raw_odcz.txt
│   │   ├── raw_pon.txt
│   │   ├── raw_styg.txt
│   │   ├── raw_szan.txt
│   │   ├── raw_vulg.txt
│   │   ├── raw_wyk.txt
│   │   └── raw_wyz.txt
│   ├── other
│   │   └── polish_stopwords.txt
│   ├── tweets_sady
│   │   ├── main
│   │   │   ├── sady_combined.csv
│   │   │   ├── sady_date_annotated.csv
│   │   │   ├── sady_infos_raw.csv
│   │   │   └── sady_infos_sanitized.csv
│   │   └── processed
│   │       ├── lemmas.csv
│   │       ├── other_scores.csv
│   │       ├── poc_scores.csv
│   │       ├── sady_duplicated.csv
│   │       └── topic_poc_scores.csv
│   ├── tweets_supplement
│   │   ├── main
│   │   │   ├── sady_2015-0405_raw.csv
│   │   │   ├── sady_2015-0405_sanitized.csv
│   │   │   ├── sady_2016-0206_raw.csv
│   │   │   └── sady_2016-0206_sanitized.csv
│   │   └── processed
│   ├── vulgars
│   │   ├── ext_vulg.txt
│   │   ├── lemm_vulg.txt
│   │   ├── polish_additionalwn_vulgars.txt
│   │   ├── polish_vulgar_phrases.txt
│   │   ├── polish_vulgar_words_github.txt
│   │   ├── polish_vulgar_words.txt
│   │   └── raw_vulg.txt
│   ├── vulgars_additionalwn
│   │   ├── ass.csv
│   │   ├── babol.csv
│   │   ├── bachnąć.csv
│   │   ├── badziewie.csv
│   │   ├── ------------
│   │   ├── zrobić_laskę.csv
│   │   ├── zrobić_loda.csv
│   │   ├── żłopanie.csv
│   │   └── żłopnąć.csv
│   ├── vulgars_net
│   │   ├── afa.csv
│   │   ├── a_gówno.csv
│   │   ├── bać_się_o_własną_dupę.csv
│   │   ├── bladź.csv
│   │   ├── ------------
│   │   ├── zjeb.csv
│   │   ├── zjeby.csv
│   │   ├── zrobić_w_chuja.csv
│   │   └── zrobić_z_dupy_garaż.csv
│   └── vulgars__texts
│       ├── afa__texts.csv
│       ├── a_gówno__texts.csv
│       ├── ass__texts.csv
│       ├── babol__texts.csv
│       ├── ------------
│       ├── zrobić_w_chuja__texts.csv
│       ├── zrobić_z_dupy_garaż__texts.csv
│       ├── żłopanie__texts.csv
│       └── żłopnąć__texts.csv
├── HSD2.0_models.zip
├── models
│   ├── lda
│   │   ├── lda_groz.pkl
│   │   ├── lda_odcz.pkl
│   │   ├── lda_pon.pkl
│   │   ├── lda_styg.pkl
│   │   ├── lda_szan.pkl
│   │   ├── lda_vulg.pkl
│   │   ├── lda_wyk.pkl
│   │   └── lda_wyz.pkl
│   ├── lexical
│   │   └── lex.pkl
│   ├── plwordnet_3_0
│   │   ├── LICENSE
│   │   ├── plwordnet-3.0-visdisc.xml
│   │   ├── plwordnet-3.0.xml
│   │   ├── readme-Eng.txt
│   │   └── readme-Pol.txt
│   └── simple_ml
│       ├── sml_DTC-entropy.pkl
│       ├── sml_DTC-gini.pkl
│       ├── sml_LRC-l1.pkl
│       ├── sml_LRC-l2.pkl
│       ├── sml_RFC-entropy.pkl
│       ├── sml_RFC-gini.pkl
│       ├── sml_SGD-l1.pkl
│       └── sml_SGD-l2.pkl
├── src
│   ├── analysis
│   │   ├── lda.py
│   │   ├── other.py
│   │   ├── poc.py
│   │   └── topic_poc.py
│   ├── classifiers
│   │   ├── Classifier.py
│   │   ├── LexicalClassifier.py
│   │   └── SimpleMLClassifier.py
│   ├── constants.py
│   ├── dataframes
│   │   ├── cards.py
│   │   ├── duplication.py
│   │   ├── timeline.py
│   │   └── utils.py
│   ├── extension
│   │   ├── com.py
│   │   ├── lemm.py
│   │   ├── sim.py
│   │   └── syn.py
│   ├── measures.py
│   ├── sanitization.py
│   ├── utils
│   │   ├── dates.py
│   │   ├── ext.py
│   │   ├── lemm.py
│   │   ├── ops.py
│   │   ├── raw.py
│   │   └── texts.py
│   ├── vectorizers
│   │   ├── CharacterVectorizer.py
│   │   └── Vectorizer.py
│   ├── visualization
│   │   ├── cards.py
│   │   ├── classification.py
│   │   └── stats.py
│   └── webscraping.py
└── tasks.odt
```

28 directories, 2648 files
