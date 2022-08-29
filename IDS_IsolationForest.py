import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from preprocessing import make_data_target

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

# Load dataset
data = pd.read_csv('dataset/Pre_train_D_attack_free.csv')
train_df = make_data_target(data)
train_df = train_df.drop(columns=['Class', 'Subclass'])
train_df.hist(bins=50, figsize=(20,20))