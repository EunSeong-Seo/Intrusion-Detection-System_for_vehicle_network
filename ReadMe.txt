


# IDS for vehicle network(Controller Area Network)

This is IDS for vehicle network.

0. You can download dataset below site
    https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset#files
    I used 0_Preliminary\0_Training\Pre_train_D_1 as train/test dataset
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
    Flooding       0.00      0.00      0.00      7704
     Fuzzing       0.00      0.00      0.00      4524
      Normal       0.91      1.00      0.95    146750
      Replay       0.00      0.00      0.00      2102
    Spoofing       0.00      0.00      0.00       198

    accuracy                           0.91    161278
   macro avg       0.18      0.20      0.19    161278
weighted avg       0.83      0.91      0.87    161278


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
    Flooding       1.00      1.00      1.00      7704
     Fuzzing       1.00      0.99      0.99      4524
      Normal       0.99      1.00      1.00    146750
      Replay       0.81      0.65      0.72      2102
    Spoofing       0.99      1.00      0.99       198

    accuracy                           0.99    161278
   macro avg       0.96      0.93      0.94    161278
weighted avg       0.99      0.99      0.99    161278



Thank you.
