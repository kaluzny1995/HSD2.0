# Project directory structure
**WARNING: Directories: charts, data and models are not in repository due to their size! Download them directly from Google Drive.**

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
├── 11. CharAndWordVectorization.ipynb
├── 12. TextVectorization.ipynb
├── 13. SimpleMLVectorClassifier.ipynb
├── 14a. SimpleMLCLassifierVectorizers (I).ipynb
├── 14b. SimpleMLCLassifierVectorizers (II).ipynb
├── 15a. DLCLassifierVectorizers (I).ipynb
├── 15b. DLCLassifierVectorizers (II).ipynb
├── 16. DLDenseNNVectorClassifier.ipynb
├── 17. DLConv1dNNVectorClassifier.ipynb
├── 18a. DLRecurrentNNVectorClassifier.ipynb
├── 18b. DLLSTMNNVectorClassifier.ipynb
├── 18c. DLGRUNNVectorClassifier.ipynb
├── 19a. DLConv1dRecurrentNNVectorClassifier.ipynb
├── 19b. DLConv1dLSTMNNVectorClassifier.ipynb
├── 19c. DLConv1dGRUNNVectorClassifier.ipynb
├── charts
│   ├── 00. schemes
│   │   ├── HSD2.0_scheme01.png
│   │   ├── HSD2.0_scheme02.png
│   │   └── HSD2.0_scheme03.png
│   ├── 01. initial_data_analysis
│   │   ├── combination_of_classes_cardinalities_upset.png
│   │   ├── empirical_sentiment_cardinalities_pie.png
│   │   ├── hateful_phrases_cardinalities_bar.png
│   │   ├── single_class_cardinalities_hbar.png
│   │   └── vulgar_words_cardinalities_venn.png
│   ├── 02. statistical_primary_analysis
│   │   ├── like_counts_hists.png
│   │   ├── replies_counts_hists.png
│   │   ├── retweet_counts_hists.png
│   │   ├── tweet_count_bars.png
│   │   ├── tweets_all_timeline.png
│   │   ├── tweets_groz_timeline.png
│   │   ├── tweets_odcz_timeline.png
│   │   ├── tweets_pon_timeline.png
│   │   ├── tweets_styg_timeline.png
│   │   ├── tweets_szan_timeline.png
│   │   ├── tweets_wyk_timeline.png
│   │   ├── tweets_wyz_timeline.png
│   │   └── tweet_yearly_counts_pie.png
│   ├── 03. extended_data_analysis
│   │   ├── combination_of_class_cardinalities_upset.png
│   │   ├── hateful_phrases_cardinalities_bar.png
│   │   ├── hateful_phrases_cardinalities_comp_bar.png
│   │   ├── single_class_cardinalities_bar.png
│   │   └── single_class_cardinalities_comp_bar.png
│   ├── 04. lexical_classifier
│   │   ├── confusion_matrices.png
│   │   ├── f_measure_lines_groz.png
│   │   ├── f_measure_lines_odcz.png
│   │   ├── f_measure_lines_pon.png
│   │   ├── f_measure_lines_styg.png
│   │   ├── f_measure_lines_szan.png
│   │   ├── f_measure_lines_wyk.png
│   │   └── f_measure_lines_wyz.png
│   ├── 05. simple_ml_classifier
│   │   ├── best_F_bars.png
│   │   ├── confusion_matrices_DTC-entropy.png
│   │   ├── confusion_matrices_DTC-gini.png
│   │   ├── confusion_matrices_LRC-l1.png
│   │   ├── confusion_matrices_LRC-l2.png
│   │   ├── confusion_matrices_RFC-entropy.png
│   │   ├── confusion_matrices_RFC-gini.png
│   │   ├── confusion_matrices_SGD-l1.png
│   │   ├── confusion_matrices_SGD-l2.png
│   │   ├── models_F_bars.png
│   │   ├── models_P_bars.png
│   │   └── models_R_bars.png
│   ├── 06. simple_ml_vector_classifier
│   │   ├── best_F_bars.png
│   │   ├── ***
│   │   └── models_R_bars.png
│   ├── 07a. simple_ml_classifier_vectors
│   │   ├── best_F_bars_2.png
│   │   ├── best_F_bars.png
│   │   ├── confusion_matrices_BERT-pret.png
│   │   ├── confusion_matrices_BERT-retr.png
│   │   ├── confusion_matrices_Chars.png
│   │   ├── confusion_matrices_FT-mtr-s.png
│   │   ├── confusion_matrices_FT-mtr-u.png
│   │   ├── confusion_matrices_FT-pret.png
│   │   ├── confusion_matrices_RoBERTa-pret.png
│   │   ├── confusion_matrices_RoBERTa-retr.png
│   │   ├── confusion_matrices_Simple-BoW.png
│   │   ├── confusion_matrices_TFIDF.png
│   │   ├── confusion_matrices_TF.png
│   │   ├── confusion_matrices_W2V-mtr-CBoW.png
│   │   ├── confusion_matrices_W2V-mtr-SkipGram.png
│   │   ├── confusion_matrices_W2V-pret-CBoW.png
│   │   ├── confusion_matrices_W2V-pret-SkipGram.png
│   │   ├── models_F_bars_2.png
│   │   ├── models_F_bars.png
│   │   ├── models_P_bars_2.png
│   │   ├── models_P_bars.png
│   │   ├── models_R_bars_2.png
│   │   └── models_R_bars.png
│   ├── 07b. dl_classifier_vectors
│   │   ├── best_F_bars_2.png
│   │   ├── ***
│   │   └── models_R_bars.png
│   ├── 08a. dl_dense_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── ***
│   │   └── w2_models_R_bars.png
│   ├── 08b. dl_conv1d_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── ***
│   │   └── w2_models_R_bars.png
│   ├── 08c. dl_recurrent_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── ***
│   │   └── w2_models_R_bars.png
│   ├── 08d. dl_lstm_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── ***
│   │   └── w2_models_R_bars.png
│   ├── 08e. dl_gru_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── ***
│   │   └── w2_models_R_bars.png
│   ├── 08f. dl_conv1d_recurrent_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── ***
│   │   └── w2_models_R_bars.png
│   ├── 08g. dl_conv1d_lstm_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── ***
│   │   └── w2_models_R_bars.png
│   └── 08h. dl_conv1d_gru_classifier
│       ├── w2_best_F_bars.png
│       ├── ***
│       └── w2_models_R_bars.png
├── data
│   ├── hateful
│   │   ├── ext_groz.txt
│   │   ├── ***
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
│   │       ├── annotation_sheet.csv
│   │       ├── annotation_sheet_empty.csv
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
│   │   ├── ***
│   │   ├── zrobić_loda.csv
│   │   ├── żłopanie.csv
│   │   └── żłopnąć.csv
│   ├── vulgars_net
│   │   ├── afa.csv
│   │   ├── a_gówno.csv
│   │   ├── bać_się_o_własną_dupę.csv
│   │   ├── ***
│   │   ├── zjeby.csv
│   │   ├── zrobić_w_chuja.csv
│   │   └── zrobić_z_dupy_garaż.csv
│   └── vulgars__texts
│       ├── afa__texts.csv
│       ├── a_gówno__texts.csv
│       ├── ass__texts.csv
│       ├── ***
│       ├── zrobić_z_dupy_garaż__texts.csv
│       ├── żłopanie__texts.csv
│       └── żłopnąć__texts.csv
├── models
│   ├── bert
│   │   ├── bert_bert
│   │   │   ├── 0_Transformer
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model.bin
│   │   │   │   ├── sentence_bert_config.json
│   │   │   │   ├── special_tokens_map.json
│   │   │   │   ├── tokenizer_config.json
│   │   │   │   └── vocab.txt
│   │   │   ├── 1_Pooling
│   │   │   │   └── config.json
│   │   │   ├── config.json
│   │   │   └── modules.json
│   │   └── bert_roberta
│   │       ├── 0_Transformer
│   │       │   ├── config.json
│   │       │   ├── merges.txt
│   │       │   ├── pytorch_model.bin
│   │       │   ├── sentence_bert_config.json
│   │       │   ├── special_tokens_map.json
│   │       │   ├── tokenizer_config.json
│   │       │   └── vocab.json
│   │       ├── 1_Pooling
│   │       │   └── config.json
│   │       ├── config.json
│   │       └── modules.json
│   ├── classification
│   │   ├── dl_vecs
│   │   │   ├── dlv_BERT-pret_dense.pkl
│   │   │   ├── ***
│   │   │   └── dlv_W2V-pret-SkipGram_dense.pt
│   │   ├── lexical
│   │   │   └── lex.pkl
│   │   ├── simple_ml
│   │   │   ├── sml_DTC-entropy.pkl
│   │   │   ├── ***
│   │   │   └── sml_SGD-l2.pkl
│   │   ├── simple_ml_vecs
│   │   │   ├── smlc_rfc_BERT-pret.pkl
│   │   │   ├── ***
│   │   │   └── smlc_rfc_W2V-pret-SkipGram.pkl
│   │   ├── simple_vec_ml
│   │   │   ├── smlv_ft_DTC-entropy.pkl
│   │   │   ├── ***
│   │   │   └── smlv_ft_SGD-l2.pkl
│   │   └── vec_dl
│   │       ├── conv1d_gru_w2
│   │       │   ├── dlvc_20-100-0.pkl
│   │       │   ├── ***
│   │       │   └── dlvc_8-50-1.pt
│   │       ├── conv1d_lstm_w2
│   │       │   ├── dlvc_20-100-0.pkl
│   │       │   ├── ***
│   │       │   └── dlvc_8-50-1.pt
│   │       ├── conv1d_recurrent_w2
│   │       │   ├── dlvc_20-100-0.pkl
│   │       │   ├── ***
│   │       │   └── dlvc_8-50-1.pt
│   │       ├── conv1d_w2
│   │       │   ├── dlvc_32-3-2.pkl
│   │       │   ├── ***
│   │       │   └── dlvc_64-5-6.pt
│   │       ├── dense_w2
│   │       │   ├── dlvc_300-0-1.pkl
│   │       │   ├── ***
│   │       │   └── dlvc_500-1-3.pt
│   │       ├── gru_w2
│   │       │   ├── dlvc_1-0-0.pkl
│   │       │   ├── ***
│   │       │   └── dlvc_5-1-1.pt
│   │       ├── lstm_w2
│   │       │   ├── dlvc_1-0-0.pkl
│   │       │   ├── ***
│   │       │   └── dlvc_5-1-1.pt
│   │       └── recurrent_w2
│   │           ├── dlvc_1-0-0.pkl
│   │           ├── ***
│   │           └── dlvc_5-1-1.pt
│   ├── fasttext
│   │   ├── ft_data_s.txt
│   │   ├── ft_data_u.txt
│   │   ├── ft_s.bin
│   │   ├── ft_u.bin
│   │   └── kgr10_orths.vec.bin
│   ├── lda
│   │   ├── lda_groz.pkl
│   │   ├── lda_odcz.pkl
│   │   ├── lda_pon.pkl
│   │   ├── lda_styg.pkl
│   │   ├── lda_szan.pkl
│   │   ├── lda_vulg.pkl
│   │   ├── lda_wyk.pkl
│   │   └── lda_wyz.pkl
│   ├── plwordnet_3_0
│   │   ├── LICENSE
│   │   ├── plwordnet-3.0-visdisc.xml
│   │   ├── plwordnet-3.0.xml
│   │   ├── readme-Eng.txt
│   │   └── readme-Pol.txt
│   ├── vectorization
│   │   ├── char
│   │   │   └── chv.pkl
│   │   ├── text
│   │   │   ├── totbertv_bert.pkl
│   │   │   ├── totbertv_roberta.pkl
│   │   │   ├── totftv_s.pkl
│   │   │   ├── totftv_u.pkl
│   │   │   ├── tptbertv_bert.pkl
│   │   │   ├── tptbertv_roberta.pkl
│   │   │   ├── tptftv.pkl
│   │   │   ├── ttfidf_tfidf.pkl
│   │   │   └── ttfidf_tf.pkl
│   │   └── word
│   │       ├── wotv_CBoW.pkl
│   │       ├── wotv_SkipGram.pkl
│   │       ├── wptv_CBoW.pkl
│   │       ├── wptv_SkipGram.pkl
│   │       └── wsbv.pkl
│   └── word2vec
│       ├── nkjp-lemmas-all-100-cbow-hs.txt.gz
│       ├── nkjp-lemmas-all-100-skipg-hs.txt.gz
│       ├── w2v_CBoW.bin
│       └── w2v_SkipGram.bin
├── src
│   ├── analysis
│   │   ├── lda.py
│   │   ├── other.py
│   │   ├── poc.py
│   │   └── topic_poc.py
│   ├── classifiers
│   │   ├── Classifier.py
│   │   ├── DLVectorClassifier.py
│   │   ├── LexicalClassifier.py
│   │   ├── SimpleMLClassifier.py
│   │   └── SimpleMLVectorClassifier.py
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
│   ├── nn
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   └── TweetsDataset.py
│   │   └── models
│   │       ├── Conv1dNet.py
│   │       ├── Conv1dRecurrentNet.py
│   │       ├── DenseNet.py
│   │       ├── __init__.py
│   │       └── RecurrentNet.py
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
│   │   ├── TextOwnTrainedBERTVectorizer.py
│   │   ├── TextOwnTrainedFTVectorizer.py
│   │   ├── TextPretrainedBERTVectorizer.py
│   │   ├── TextPretrainedFTVectorizer.py
│   │   ├── TextTFIDFVectorizer.py
│   │   ├── Vectorizer.py
│   │   ├── WordOwnTrainedVectorizer.py
│   │   ├── WordPretrainedVectorizer.py
│   │   └── WordSimpleBoWVectorizer.py
│   ├── visualization
│   │   ├── cards.py
│   │   ├── classification.py
│   │   └── stats.py
│   └── webscraping.py
└── tasks.odt
```

83 directories, 3264 files
