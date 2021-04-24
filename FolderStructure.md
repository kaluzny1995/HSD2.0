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
├── 20a. DLClassifierHyperparamsDense.ipynb
├── 20b. DLClassifierHyperparams1dConv.ipynb
├── 20c. DLClassifierHyperparamsLSTM.ipynb
├── 20d. DLClassifierHyperparamsGRU.ipynb
├── 20e. DLClassifierHyperparams1dConvLSTM.ipynb
├── 20f. DLClassifierHyperparams1dConvGRU.ipynb
├── 21. BestModels.ipynb
├── 22. HateSpeechPrediction.ipynb
├── 23. StatisticalSecondaryDataAnalysis.ipynb
├── 24. ErrorAnalysis.ipynb
├── 25a. Experiment E1.ipynb
├── 25b. Experiment E2.ipynb
├── 25c. Experiment E3.ipynb
├── 25d. Experiment E4.ipynb
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
│   │   ├── all_ym_tweet_amounts_lines.png
│   │   ├── all_ym_tweet_wordcounts_lines.png
│   │   ├── like_counts_hists.png
│   │   ├── replies_counts_hists.png
│   │   ├── retweet_counts_hists.png
│   │   ├── tweet_count_bars.png
│   │   ├── tweets_groz_timeline.png
│   │   ├── tweets_odcz_timeline.png
│   │   ├── tweets_pon_timeline.png
│   │   ├── tweets_styg_timeline.png
│   │   ├── tweets_szan_timeline.png
│   │   ├── tweets_timeline.png
│   │   ├── tweets_wyk_timeline.png
│   │   ├── tweets_wyz_timeline.png
│   │   ├── tweet_yearly_counts_pie.png
│   │   └── tweet_ym_amounts_lines.png
│   ├── 03. extended_data_analysis
│   │   ├── combination_of_class_cardinalities_upset.png
│   │   ├── hateful_phrases_cardinalities_bar.png
│   │   ├── hateful_phrases_cardinalities_comp_bar.png
│   │   ├── single_class_cardinalities_bar.png
│   │   └── single_class_cardinalities_comp_bar.png
│   ├── 04. lexical_classifier
│   │   ├── confusion_matrices.png
│   │   ├── f_measure_lines_groz.png
│   │   ├── ...
│   │   └── f_measure_lines_wyz.png
│   ├── 05. simple_ml_classifier
│   │   ├── best_F_bars.png
│   │   ├── confusion_matrices_DTC-entropy.png
│   │   ├── ...
│   │   ├── confusion_matrices_SGD-l2.png
│   │   ├── models_F_bars.png
│   │   ├── models_P_bars.png
│   │   └── models_R_bars.png
│   ├── 06. simple_ml_vector_classifier
│   │   ├── best_F_bars.png
│   │   ├── confusion_matrices_DTC-entropy.png
│   │   ├── ...
│   │   ├── confusion_matrices_SGD-l2.png
│   │   ├── models_F_bars.png
│   │   ├── models_P_bars.png
│   │   └── models_R_bars.png
│   ├── 07a. simple_ml_classifier_vectors
│   │   ├── best_F_bars_2.png
│   │   ├── best_F_bars.png
│   │   ├── confusion_matrices_BERT-pret.png
│   │   ├── ...
│   │   ├── confusion_matrices_W2V-pret-SkipGram.png
│   │   ├── models_F_bars_2.png
│   │   ├── models_F_bars.png
│   │   ├── models_P_bars_2.png
│   │   ├── models_P_bars.png
│   │   ├── models_R_bars_2.png
│   │   └── models_R_bars.png
│   ├── 07b. dl_classifier_vectors
│   │   ├── best_F_bars_2.png
│   │   ├── best_F_bars.png
│   │   ├── best_history_lines_2.png
│   │   ├── best_history_lines.png
│   │   ├── confusion_matrices_BERT-pret.png
│   │   ├── ...
│   │   ├── confusion_matrices_W2V-pret-SkipGram.png
│   │   ├── models_F_bars_2.png
│   │   ├── models_F_bars.png
│   │   ├── models_P_bars_2.png
│   │   ├── models_P_bars.png
│   │   ├── models_R_bars_2.png
│   │   └── models_R_bars.png
│   ├── 08a. dl_dense_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_300-0-1.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_500-1-3.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 08b. dl_conv1d_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_32-3-1.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_64-5-6.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 08c. dl_recurrent_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_1-0-0.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_5-1-1.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 08d. dl_lstm_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_1-0-0.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_5-1-1.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 08e. dl_gru_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_1-0-0.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_5-1-1.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 08f. dl_conv1d_recurrent_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_20-100-0.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_8-50-1.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 08g. dl_conv1d_lstm_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_20-100-0.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_8-50-1.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 08h. dl_conv1d_gru_classifier
│   │   ├── w2_best_F_bars.png
│   │   ├── w2_best_history_lines.png
│   │   ├── w2_confusion_matrices_20-100-0.png
│   │   ├── ...
│   │   ├── w2_confusion_matrices_8-50-1.png
│   │   ├── w2_models_F_bars.png
│   │   ├── w2_models_P_bars.png
│   │   └── w2_models_R_bars.png
│   ├── 09. dl_hparams
│   │   ├── 1dcgru_w2_best_F_bars.png
│   │   ├── 1dcgru_w2_best_history_lines.png
│   │   ├── 1dcgru_w2_confusion_matrices_adamw-ams-cyc.png
│   │   ├── ...
│   │   ├── 1dcgru_w2_confusion_matrices_sgd-rop.png
│   │   ├── 1dcgru_w2_models_F_bars.png
│   │   ├── 1dcgru_w2_models_P_bars.png
│   │   ├── 1dcgru_w2_models_R_bars.png
│   │   ├── 1dclstm_w2_best_F_bars.png
│   │   ├── 1dclstm_w2_best_history_lines.png
│   │   ├── 1dclstm_w2_confusion_matrices_adamw-ams-cyc.png
│   │   ├── ...
│   │   ├── 1dclstm_w2_confusion_matrices_sgd-rop.png
│   │   ├── 1dclstm_w2_models_F_bars.png
│   │   ├── 1dclstm_w2_models_P_bars.png
│   │   ├── 1dclstm_w2_models_R_bars.png
│   │   ├── c1d_w2_best_F_bars.png
│   │   ├── c1d_w2_best_history_lines.png
│   │   ├── c1d_w2_confusion_matrices_adamw-ams-cyc.png
│   │   ├── ...
│   │   ├── c1d_w2_confusion_matrices_sgd-rop.png
│   │   ├── c1d_w2_models_F_bars.png
│   │   ├── c1d_w2_models_P_bars.png
│   │   ├── c1d_w2_models_R_bars.png
│   │   ├── dense_w2_best_F_bars.png
│   │   ├── dense_w2_best_history_lines.png
│   │   ├── dense_w2_confusion_matrices_adamw-ams-cyc.png
│   │   ├── ...
│   │   ├── dense_w2_confusion_matrices_sgd-rop.png
│   │   ├── dense_w2_models_F_bars.png
│   │   ├── dense_w2_models_P_bars.png
│   │   ├── dense_w2_models_R_bars.png
│   │   ├── gru_w2_best_F_bars.png
│   │   ├── gru_w2_best_history_lines.png
│   │   ├── gru_w2_confusion_matrices_adamw-ams-cyc.png
│   │   ├── ...
│   │   ├── gru_w2_confusion_matrices_sgd-rop.png
│   │   ├── gru_w2_models_F_bars.png
│   │   ├── gru_w2_models_P_bars.png
│   │   ├── gru_w2_models_R_bars.png
│   │   ├── lstm_w2_best_F_bars.png
│   │   ├── lstm_w2_best_history_lines.png
│   │   ├── lstm_w2_confusion_matrices_adamw-ams-cyc.png
│   │   ├── ...
│   │   ├── lstm_w2_confusion_matrices_sgd-rop.png
│   │   ├── lstm_w2_models_F_bars.png
│   │   ├── lstm_w2_models_P_bars.png
│   │   └── lstm_w2_models_R_bars.png
│   ├── 10. best_models
│   │   ├── best_F_bars.png
│   │   ├── confusion_matrices_1dCNN+GRU-HP.png
│   │   ├── ...
│   │   ├── confusion_matrices_SGDVC.png
│   │   ├── deep_models_valid_A_bars.png
│   │   ├── deep_models_valid_F_bars.png
│   │   ├── models_F_bars.png
│   │   ├── models_P_bars.png
│   │   └── models_R_bars.png
│   ├── 11. statistical_secondary_analysis
│   │   ├── all_ym_tweet_amounts_lines.png
│   │   ├── all_ym_tweet_wordcounts_lines.png
│   │   ├── like_counts_hists.png
│   │   ├── replies_counts_hists.png
│   │   ├── retweet_counts_hists.png
│   │   ├── tweet_count_bars.png
│   │   ├── tweets_groz_timeline.png
│   │   ├── tweets_odcz_timeline.png
│   │   ├── tweets_pon_timeline.png
│   │   ├── tweets_styg_timeline.png
│   │   ├── tweets_szan_timeline.png
│   │   ├── tweets_timeline.png
│   │   ├── tweets_wyk_timeline.png
│   │   ├── tweets_wyz_timeline.png
│   │   ├── tweet_yearly_counts_pie.png
│   │   └── tweet_ym_amounts_lines.png
│   └── 12. experiments
│       ├── E1_a_lines.png
│       ├── ...
│       ├── E1_wyz_lines.png
│       ├── E2_a_lines.png
│       ├── ...
│       ├── E2_wyz_lines.png
│       ├── E3_a_lines.png
│       ├── ...
│       ├── E3_wyz_lines.png
│       ├── E4_a_lines.png
│       ├── ...
│       └── E4_wyz_lines.png
├── data
│   ├── hateful
│   │   ├── ext_groz.txt
│   │   ├── ...
│   │   ├── ext_wyz.txt
│   │   ├── lemm_groz.txt
│   │   ├── ...
│   │   ├── lemm_wyz.txt
│   │   ├── raw_groz.txt
│   │   ├── ...
│   │   └── raw_wyz.txt
│   ├── other
│   │   └── polish_stopwords.txt
│   ├── results
│   │   ├── best_models_results.csv
│   │   ├── best_models_train_results.csv
│   │   ├── experiment_E1_results.csv
│   │   ├── experiment_E2_results.csv
│   │   ├── experiment_E3_results.csv
│   │   ├── experiment_E4_results.csv
│   │   ├── predictions_1dCNN+GRU-HP.csv
│   │   ├── predictions_RNN.csv
│   │   └── predictions_SGDVC.csv
│   ├── tweets_2014_2020
│   │   ├── all_lemmas.csv
│   │   ├── all_other_scores.csv
│   │   ├── all_poc_scores.csv
│   │   ├── all_topic_poc_scores.csv
│   │   └── sady_all_sanitized.csv
│   ├── tweets_sady
│   │   ├── main
│   │   │   ├── sady_combined.csv
│   │   │   ├── sady_combined_testonly.csv
│   │   │   ├── sady_date_annotated.csv
│   │   │   ├── sady_date_annotated_testonly.csv
│   │   │   ├── sady_infos_raw.csv
│   │   │   └── sady_infos_sanitized.csv
│   │   └── processed
│   │       ├── annotation_sheet_a0.csv
│   │       ├── annotation_sheet_a1.csv
│   │       ├── annotation_sheet_empty.csv
│   │       ├── lemmas.csv
│   │       ├── lemmas_testonly.csv
│   │       ├── other_scores.csv
│   │       ├── other_scores_testonly.csv
│   │       ├── poc_scores.csv
│   │       ├── poc_scores_testonly.csv
│   │       ├── sady_duplicated.csv
│   │       ├── topic_poc_scores.csv
│   │       └── topic_poc_scores_testonly.csv
│   ├── tweets_supplement
│   │   ├── sady_2015-0405_raw.csv
│   │   ├── ...
│   │   ├── sady_2019-1203_raw.csv
│   │   ├── sady_supplement_raw.csv
│   │   └── sady_supplement_sanitized.csv
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
│   │   ├── ...
│   │   ├── zrobić_loda.csv
│   │   ├── żłopanie.csv
│   │   └── żłopnąć.csv
│   ├── vulgars_net
│   │   ├── afa.csv
│   │   ├── a_gówno.csv
│   │   ├── bać_się_o_własną_dupę.csv
│   │   ├── ...
│   │   ├── zjeby.csv
│   │   ├── zrobić_w_chuja.csv
│   │   └── zrobić_z_dupy_garaż.csv
│   └── vulgars__texts
│       ├── afa__texts.csv
│       ├── a_gówno__texts.csv
│       ├── ass__texts.csv
│       ├── ...
│       ├── zrobić_z_dupy_garaż__texts.csv
│       ├── żłopanie__texts.csv
│       └── żłopnąć__texts.csv
├── HSD2.0_charts.zip
├── HSD2.0_data.zip
├── HSD2.0_models.zip
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
│   │   │   ├── dlv_BERT-pret_dense.pt
│   │   │   ├── ...
│   │   │   ├── dlv_W2V-pret-SkipGram_dense.pkl
│   │   │   └── dlv_W2V-pret-SkipGram_dense.pt
│   │   ├── lexical
│   │   │   └── lex.pkl
│   │   ├── simple_ml
│   │   │   ├── sml_DTC-entropy.pkl
│   │   │   ├── ...
│   │   │   └── sml_SGD-l2.pkl
│   │   ├── simple_ml_vecs
│   │   │   ├── smlc_rfc_BERT-pret.pkl
│   │   │   ├── ...
│   │   │   └── smlc_rfc_W2V-pret-SkipGram.pkl
│   │   ├── simple_vec_ml
│   │   │   ├── smlv_ft_DTC-entropy.pkl
│   │   │   ├── ...
│   │   │   └── smlv_ft_SGD-l2.pkl
│   │   └── vec_dl
│   │       ├── conv1d_gru_w2
│   │       │   ├── dlvc_20-100-0.pkl
│   │       │   ├── dlvc_20-100-0.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_8-50-1.pkl
│   │       │   └── dlvc_8-50-1.pt
│   │       ├── conv1d_lstm_w2
│   │       │   ├── dlvc_20-100-0.pkl
│   │       │   ├── dlvc_20-100-0.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_8-50-1.pkl
│   │       │   └── dlvc_8-50-1.pt
│   │       ├── conv1d_recurrent_w2
│   │       │   ├── dlvc_20-100-0.pkl
│   │       │   ├── dlvc_20-100-0.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_8-50-1.pkl
│   │       │   └── dlvc_8-50-1.pt
│   │       ├── conv1d_w2
│   │       │   ├── dlvc_32-3-1.pkl
│   │       │   ├── dlvc_32-3-1.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_64-5-4.pkl
│   │       │   └── dlvc_64-5-4.pt
│   │       ├── dense_w2
│   │       │   ├── dlvc_300-0-1.pkl
│   │       │   ├── dlvc_300-0-1.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_500-1-3.pkl
│   │       │   └── dlvc_500-1-3.pt
│   │       ├── gru_w2
│   │       │   ├── dlvc_1-0-0.pkl
│   │       │   ├── dlvc_1-0-0.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_5-1-1.pkl
│   │       │   └── dlvc_5-1-1.pt
│   │       ├── hparams_w2
│   │       │   ├── dlvc_1dcgru_adamw-ams-cyc.pkl
│   │       │   ├── dlvc_1dcgru_adamw-ams-cyc.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_lstm_sgd-rop.pkl
│   │       │   └── dlvc_lstm_sgd-rop.pt
│   │       ├── lstm_w2
│   │       │   ├── dlvc_1-0-0.pkl
│   │       │   ├── dlvc_1-0-0.pt
│   │       │   ├── ...
│   │       │   ├── dlvc_5-1-1.pkl
│   │       │   └── dlvc_5-1-1.pt
│   │       └── recurrent_w2
│   │           ├── dlvc_1-0-0.pkl
│   │           ├── dlvc_1-0-0.pt
│   │           ├── ...
│   │           ├── dlvc_5-1-1.pkl
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
│   │       ├── wotv_CBoW-pca.pkl
│   │       ├── wotv_CBoW.pkl
│   │       ├── wotv_SkipGram-pca.pkl
│   │       ├── wotv_SkipGram.pkl
│   │       ├── wptv_CBoW-pca.pkl
│   │       ├── wptv_CBoW.pkl
│   │       ├── wptv_SkipGram-pca.pkl
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
│   ├── annotation.py
│   ├── best_models.py
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
│   ├── error_analysis.py
│   ├── experiments.py
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

88 directories, 3668 files
