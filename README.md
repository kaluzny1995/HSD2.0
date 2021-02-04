# Hate Speech Detector v2.0
---
This is a project of a multilabel classifier of tweet texts whether they contain hate speech or not. The project is being made as a Master of Science degree thesis. There are seven different types of hate speech: **abusement** *(pol. wyzywanie)*, **threatening** *(pol. grożenie)*, **exclusion** *(pol. wykluczanie)*, **dehumanization** *(pol. odczłowieczanie)*, **humiliation** *(pol. poniżanie)*, **labeling** *(pol. stygmatyzacja)* and **blackmail** *(pol. szantaż)*.

# Project directory structure
**WARNING: Not all files or directories are in repository due to their size!**

<details><summary>View directory structure</summary>
  ├── data<br />
  │   ├── sady_main<br />
  │   │   ├── sady_infos_raw.csv<br />
  │   │   ├── sady_infos_sanitized.csv<br />
  │   │   ├── sady_date_annotated.csv<br />
  │   │   ├── sady_other_scores.csv<br />
  │   │   ├── sady_pac_scores.csv<br />
  │   │   ├── sady_combined.csv<br />
  │   │   ├── sady_topic_pac_scores.csv<br />
  │   │   └── sady_simple_ml_classifier.csv<br />
  │   ├── sady_main
  │   │   ├── sady_2017_0105_raw.csv
  │   │   ├── sady_2017_0105_raw_pl.csv
  │   │   ├── sady_2017_0105_part_sanitized.csv
  │   │   └── sady_2017_0105_sanitized.csv
  ├── models
  │   ├── lda
  │   │   ├── lda_wyz.pkl
  │   │   ├── lda_groz.pkl
  │   │   ├── lda_wyk.pkl
  │   │   ├── lda_odcz.pkl
  │   │   ├── lda_pon.pkl
  │   │   ├── lda_styg.pkl
  │   │   ├── lda_szan.pkl
  │   │   └── lda_vulg.pkl
  │   ├── plwordnet_3_0
  │   │   ├── LICENSE
  │   │   ├── plwordnet-3.0.xml
  │   │   ├── plwordnet-3.0-visdic.xml
  │   │   ├── readme-Eng.txt
  │   │   └── readme-Pol.txt
  │   ├── simple_ml
  │   │   ├── DT_entropy.pkl
  │   │   ├── DT_gini.pkl
  │   │   ├── RF_entropy_balanced.pkl
  │   │   ├── RF_entropy_balanced_subsample.pkl
  │   │   ├── RF_gini_balanced.pkl
  │   │   ├── RF_gini_balanced_subsample.pkl
  │   │   ├── SV_linear_1_0.pkl
  │   │   ├── SV_poly_3_1_0.pkl
  │   │   └── SV_poly_5_1_0.pkl
  ├── charts
  │   ├── initial_data_analysis
  │   │   ├── cardinalities.png
  │   │   └── percentages.png
  |   ├── lexical_classifier
  │   │   ├── acc_fa_wyz.png
  │   │   ├── acc_fa_groz.png
  │   │   ├── acc_fa_wyk.png
  │   │   ├── acc_fa_odcz.png
  │   │   ├── acc_fa_pon.png
  │   │   ├── acc_fa_styg.png
  │   │   ├── acc_fa_szan.png
  │   │   └── conf_matrices.png
  │   └── simple_ml_classifier
  │   │   ├── cms_DT_entropy.png
  │   │   ├── cms_DT_gini.png
  │   │   ├── cms_RF_entropy_balanced.png
  │   │   ├── cms_RF_entropy_balanced_subsample.png
  │   │   ├── cms_RF_gini_balanced.png
  │   │   ├── cms_RF_gini_balanced_subsample.png
  │   │   ├── cms_SV_linear_1_0.png
  │   │   ├── cms_SV_poly_3_1_0.png
  │   │   ├── cms_SV_poly_5_1_0.png
  │   │   ├── best_model.png
  │   │   └── model_acc_f1.png
  ├── WebScraping.ipynb
  ├── TweetSanitizer.ipynb
  ├── AnnotatedDataAnalysis.ipynb
  ├── VulgarPhrasesDict.ipynb
  ├── DataDuplicator.ipynb
  ├── InitialDataAnalysis.ipynb
  ├── LexicalClassifier.ipynb
  ├── SimpleMLClassifier.ipynb
  └── README.md
</details>
