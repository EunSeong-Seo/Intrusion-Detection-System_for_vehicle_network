


# IDS for vehicle network(Controller Area Network)

This is IDS for vehicle network.

0. You can download dataset below site
    https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset#files
    I used 0_Preliminary\0_Training\Pre_train_D_1 as train dataset
    and 0_Preliminary\0_Training\Pre_train_D_2 are used as test dataset
1. CAN data EDA - data_EDA.ipnb
2. Preprocess for apply machine learning models - preprocessing.py
3. Models
=========Done=========
- Logistic Regression
- RandomForest
- Decision Trees
=======Preparing=======
- Naive Bayes
- KNN
- SVM
- ANN(2 hidden layers)
- LSTM(256 hidden units)
- CNN

4. Results
4-(1) Logistic Regression
    -> worst model in my experiment

              precision    recall  f1-score   support
    Flooding       0.00      0.00      0.00     38852
     Fuzzing       0.00      0.00      0.00     22854
      Normal       0.91      1.00      0.95    811532
      Replay       0.00      0.00      0.00     13266
    Spoofing       0.00      0.00      0.00      2891

    accuracy                           0.91    889395
   macro avg       0.18      0.20      0.19    889395
weighted avg       0.83      0.91      0.87    889395


4-(2) RandomForest

              precision    recall  f1-score   support
    Flooding       1.00      1.00      1.00     38852
     Fuzzing       0.93      1.00      0.97     22854
      Normal       0.99      0.77      0.87    811532
      Replay       0.04      0.66      0.08     13266
    Spoofing       1.00      1.00      1.00      2891

    accuracy                           0.78    889395
   macro avg       0.79      0.88      0.78    889395
weighted avg       0.98      0.78      0.86    889395


4-(3) Decision Trees

              precision    recall  f1-score   support
    Flooding       1.00      1.00      1.00     38852
     Fuzzing       0.70      0.99      0.82     22854
      Normal       0.99      0.76      0.86    811532
      Replay       0.05      0.66      0.08     13266
    Spoofing       1.00      1.00      1.00      2891

    accuracy                           0.78    889395
   macro avg       0.75      0.88      0.75    889395
weighted avg       0.97      0.78      0.85    889395




Thank you.
