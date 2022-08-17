import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from preprocessing import make_train_xy_df
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# set data paths
can_dataset_path_for_D_1 = 'dataset/Pre_train_D_attack_1.csv'
can_dataset_path_for_D_2 = 'dataset/Pre_train_D_attack_2.csv'

# Load dataset
x, y = make_train_xy_df(can_dataset_path_for_D_1)

log_reg = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size= 0.2)
log_reg.fit(X_train,Y_train)
Y_pred = log_reg.predict(X_test)
print(metrics.accuracy_score(Y_test,Y_pred))