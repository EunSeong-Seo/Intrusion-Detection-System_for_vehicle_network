import pandas as pd
import warnings
from preprocessing import make_data_target
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# set data paths
can_dataset_path_for_D_1 = 'dataset/Pre_train_D_attack_1.csv'
can_dataset_path_for_D_2 = 'dataset/Pre_train_D_attack_2.csv'

# Load dataset
data, target = make_data_target(can_dataset_path_for_D_1)
X_train, X_val, Y_train, Y_val = train_test_split(data, target, test_size=0.2, stratify=target, random_state=11)

# Use Decision Tree
DT = DecisionTreeClassifier()
DT = DT.fit(X_train,Y_train)
Y_pred = DT.predict(X_val)

print(metrics.accuracy_score(Y_val, Y_pred))
print(metrics.confusion_matrix(Y_val, Y_pred))
print(metrics.classification_report(Y_val, Y_pred))