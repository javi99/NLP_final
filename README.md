# Final Project - Advanced Methods in Natural Language Processing

## Description & Motivation
Our project utilizes a combination of three powerful machine learning models - Logistic Regression, a Recurrent Neural Network (RNN) and BERT - to classify text. Logistic Regression is used as a baseline model, while the RNN helps capture sequential information in the text. BERT, a pre-trained transformer-based model, is used for fine-tuning and improving the accuracy of the model. By leveraging the strengths of these models, we can effectively classify text and achieve high accuracy in various applications such as sentiment analysis, spam detection, and more. After the implementation of these methods, we spent a considerable amount of time to scrutinize our models and find potential biases. 

## Table of Contents 
- [Data](#data)
- [Logistic Regression - Baseline](#logisticregression-baseline)
- [Recurent Neural Network](#recurentneuralnetwork)
- [BERT](#bert)
- [Understanding Models & Biases](#understandingmodels&biases)


## Data
Use a dataset from huggingface.com called DBPedia which contains the title and body of Wikipedia articles that are classified into 14 different topics. Namely, *Company*, *EducationalInstitution*, *Artist*, *Athlete*, *OfficeHoldern*, *MeanOfTransportation*, *Building*, *NaturalPlace*, *Village*, *Animal*, *Plant*, *Album*, *Film*, and *WrittenWork*.

In our first step we clean the dataset which means eliminating all articles with non English articles. This reduced the data set by a third, but also creates some class imabalances. Second, we restrict the number of words per artible to the first 80 to ensure the dimensions fit later on the modelling stage. The next steps depend on the model and will be discussed in detail in the according section.


## Logistic Regression - Baseline
As a second preprocessing stage we to *sklearn's* CountVectoriser. The CountVectorizer is a tool used for feature extraction that converts (tokensization) the text data into a matrix of word counts. The Logistic Regression model is then applied to this matrix to classify the text into different categories based on the features (word frequency in topics) extracted.

We evaluate the evolution of model performance as a function of the sample size. With only 10% of the data, the model already achieves a 96% Accuracy and F1 score which rise to 98% each as the full model is fed.


## Recurent Neural Network
As before we Tokenize and do some common pre-processing steps. However, to fit a neural network we need the OneHotEncoder. This is a technique used to convert categorical data into numerical data that can be processed by machine learning algorithms. Finally, word embeddings such as Word2Vec are a popular technique used to represent words as vectors, capturing the semantic relationships between them. By applying these techniques to preprocess text data, we can improve the accuracy of models used for tasks such as sentiment analysis, text classification, and information retrieval.

RNNs process sequential data using feedback connections to maintain information from previous time steps. They're ideal for language modeling, speech recognition, and time series prediction. At each step, RNNs produce an output and hidden state, which becomes an additional input for the next step. By updating the hidden state based on current and previous inputs, RNNs capture long-term dependencies and generate contextually relevant outputs.

Our RNN out performs the Baseline model with an accuracy and F1 score of 98.5% percet. Especially the *Plant* and *Village* classes are predicted excepionally well with a near perfect accuarcy and F1 score.


## BERT
BERT is a neural network architecture that can be fine-tuned for a wide range of NLP tasks, especially text classification. It is based on the transformer architecture and uses a bidirectional approach to pre-train contextualized word embeddings. This allows BERT to better understand the nuances of language and perform well on various NLP tasks without the need for extensive task-specific training. Data must be tokenized into subword units which are a combination of full words and subwords. Fine-tuning BERT on a labeled text classification dataset involves updating the pre-trained model's weights based on the labeled data. During training, the labeled data is fed into the model, and the model adjusts its weights to minimize the difference between its predicted outputs and the correct labels. Once the model is trained on the labeled data, it can be used to classify new, unlabeled text by feeding it into the fine-tuned BERT model.

Our BERT model slightly outperform the RNN and has a perfect recall score. Moreover, as the batches are fed we make sure to give it similar numbers of each class. We thus ensure to mitigate class imbalances before they even arise.


## Understanding Models & Biases
A way to understand our respective models (and perhaps biases) is LIME. LIME uses an example to demonstrate how the pretrained model makes it's descion. Take the **Baseline Model** as an example:
![Screenshot](screenshot.png)



