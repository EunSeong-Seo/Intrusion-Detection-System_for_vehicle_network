import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from preprocessing import make_data_target
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

# set data paths
can_dataset_path_for_D_1 = 'dataset/Pre_train_D_attack_1.csv'

# Load dataset
data, target = make_data_target(can_dataset_path_for_D_1)
X_train, X_val, Y_train, Y_val = train_test_split(data, target, test_size=0.2, stratify=target, random_state=11)

# Find optimal max_depth

cv = KFold(n_splits=5)  # Desired number of Cross Validation folds  #n_splits값이 클수록 오래걸림
accuracies = list()
max_attributes = X_val.shape[1]
depth_range = range(1, max_attributes)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    rand_clf = RandomForestClassifier(max_depth=depth)
    # print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(X_train):
        f_train = X_train.loc[train_fold]  # Extract train data with cv indices
        f_valid = X_train.loc[valid_fold]  # Extract valid data with cv indices

        model = rand_clf.fit(X_train, Y_train)
        valid_acc = model.score(X_val, Y_val)
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)
    print("Accuracy per fold: ", fold_accuracy, "\n")
    print("Average accuracy: ", avg)
    print("\n")

# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))


# Use Random Forest
clf = RandomForestClassifier(criterion='entropy', bootstrap=True, max_depth=5, random_state=42)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_val)

print(metrics.accuracy_score(Y_val, Y_pred))
print(metrics.confusion_matrix(Y_val, Y_pred))
print(metrics.classification_report(Y_val, Y_pred))

import seaborn as sns
import numpy as np
print('Feature importances : \n{}'.format(np.round(rand_clf.feature_importances_,3)))

features = ['Timestamp', 'Arbitration_ID', 'DLC','D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
for name, value in zip(features, rand_clf.feature_importances_):
    print('{} : {1: .3f}'.format(name,value))

sns.barplot(x=rand_clf.feature_importances_, y= features)