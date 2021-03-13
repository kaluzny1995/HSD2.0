# Hate Speech Detector v2.0
---
#### Description
This is a project of a multilabel classifier of polish tweet texts whether they contain hate speech or not. The project is being made as a Master of Science degree thesis. There are seven different types of hate speech: **abusement** *(pol. wyzywanie)*, **threatening** *(pol. grożenie)*, **exclusion** *(pol. wykluczanie)*, **dehumanization** *(pol. odczłowieczanie)*, **humiliation** *(pol. poniżanie)*, **labeling** *(pol. stygmatyzacja)* and **blackmail** *(pol. szantaż)*. The topics of all tweets is the attitude of people (Twitter users) to polish jurisdiction institutions. The annotated data concern the time from June 2014 to December 2016.
#### Data, models and visualization
All scraped and processed tweet data, models and chart visualizations are available here: [HSD2.0_data.zip](https://drive.google.com/file/d/1Cg1mulD2AAp7jiQ1ShZp8xgRtBODiLVg/view?usp=sharing), [HSD2.0_models.zip](https://drive.google.com/file/d/19QzvSJQ1n663e2MAV_AtZtZSs-Ts9vJF/view?usp=sharing) and [HSD2.0_charts.zip](https://drive.google.com/file/d/1mErHvy8aLiFhXP6NBCNQ_ZNRuOYRzDQ1/view?usp=sharing). Download above components and unzip them into project main directory.
#### Abbrevations
* ML - machine learning
* DL - deep learning
* NN - neural network
* DNN - deep neural network
* 1dCNN - 1-dimensional convolutional neural network
* RNN - recurrent neural network
* LSTM - long short-term memory
* GRU - gated recurrent unit

## Project stages
---
1. Webscraping
2. Tweet sanitization
3. Initial data analysis
4. Statistical primary data analysis
5. Data duplication
6. Hateful and vulgar phrases extension
7. Extended data analysis
8. Advanced data analysis
9. Lexical classifier
10. Simple ML classifier
11. Vectorization
    1. Character and word vectorization
    2. Text vectorization
12. Simple ML vector classifier
13. Simple ML classification vectorizers analysis
14. DL classification vectorizers analysis
15. DL classifiers (NN params analysis)
    1. DNN
    2. 1dCNN
    3. Simple, LSTM and GRU RNNs
    4. Siamese 1dCNN with simple, LSTM and GRU RNNs
16. DL classifier hyperparams analysis *(coming soon)*
17. Hate speech prediction with best model *(coming soon)*
18. Statistical secondary data analysis *(coming soon)*

## Webscraping
---
Scraping (i.e. downloading) raw tweet data from Twitter in topic of 'polish peoples attitude to jurisdiction institutions'. Data are scraped from certain period of time. Data scraped from June 2014 to December 2016 were multilabel annotated.

## Tweet sanitization
---
Elimination of: URL adresses, links, hashtag hash signs (#), user mentions, non-utf8 characters and redundant spaces. Separation of emoji and emoticons from tweets.

## Initial data analysis
---
Tweet class and hateful phrases cardinalities analysis with charts visualisation. There are single class cardinality and combination-of-classes cardinality. Vulgars phrases analysis with charts visualization. Empirically annotated sentiment countity analysis with visualization. Annotators agreement calculation.

## Statistical primary data analysis
---
Annoteted tweets countity timeline analysis according to certain hate-speech type. Tweets popularity analysis. Histograms of tweets popularity according to certain hate-speech type.

## Data duplication
---
Duplication of data which combination-of-classes cardinality is too low to perform stratified k-fold cross-validation.

## Hateful and vulgar phrases extension
---
Methods of phrases dictionaries extension. Additional phrases are being fetched from similar phrase vectors and synonymic wordnet phrases.

## Extended data analysis
---
All analyses like in initial data analysis, but the annotator agreement. Countity and quality analyses are being performed for duplicated data and extended phrases.

## Advanced data analysis
---
Data analysis useful for lexical classifier. Phrases Occurence Coefficient (POC) calculation for phrase denoting certain hate-type. POC are also calculated for topics detected by LDA (Latent Dirichlet Allocation) algorithm. At the and the special text metrics are calculated. These metrics are following: number of characters, number of syllables, number of words.

## Lexical classification
---
First classifier. Classifier based on POC features for hateful phrases extracted in previous task (ie. lexical features).

## Simple ML classification
---
Simple ML classifier (i.e. DecisionTree, RandomForest, etc.). Classifier based on all features extracted in advanced data analysis.

## Vectorization
---
Methods of text vectorization. There are three main levels of vectorization: character, word and sentence/text. There were analized: one character, five word and eight text level vectorizers.

## Simple ML vector classification
---
Simple ML classifier for vectorized tweets.

## Simple ML classification vectorizers analysis
---
Best vectorization model determination by vectorized tweets classification with RandomForest classifier model.

## DL classification vectorizer analysis
---
Best vectorization model determination by vectorized tweets classification with DL DNN model.

## DL classification (NN parametrs analysis)
---
Parameter analysis for various NNs: DNN, 1dCNN, RNN, LSTM, GRU and siamese 1dCNN-RNN, 1dCNN-LSTM, 1dCNN-GRU. Each DL classifier was trained with application of pretrained Word2Vec vectorization model. All analyses (with best models) were appropriately visualized. Models were trained on 20 epochs and saved after first 10.
There were analized numbers of hidden, convolutional and recurrent layers with their sizes, dropout probs and methods of dataflow (convolutional kernel dimensions or recurrent bidirectioning).

## DL classification (NN hyperparameters analysis)
---
Hyperparameters analysis for the best NN models trained in previous task. There were analized various: loss functions, optimizers with parameters, optimizer schedulers with parameters, regularization parameters. Number of epoch were increased for 20 to 50.

## Hate speech prediction with best model
---
Hate speech predictions for unannotated tweets using the best trained model.

## Statistical secondary data analysis
---
All data (ie. annotated and unannotated/predicted) tweets countity timeline analysis. All analyses like in primary statistical analysis.
