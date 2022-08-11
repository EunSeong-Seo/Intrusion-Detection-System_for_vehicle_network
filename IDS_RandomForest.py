import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# set data paths
path = can_dataset_path_for_D_1 = 'dataset/Pre_train_D_attack_1.csv'
can_dataset_path_for_D_2 = 'dataset/Pre_train_D_attack_2.csv'


# split train_x, train_y , test_x, test_y
# train, val = train_test_split(data_D_attack_train, train_size=0.80, test_size=0.20, stratify=train, shuffle = False)
def byte_encoding

# v1.0 throw IDs away
def make_train_xy_df(path):
    # Load data from path
    df = pd.read_csv(path)
    features = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'Class', 'SubClass']
    df_data = df['Data'].str.split(' ', expand=True)
    df_data.columns = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
    df_data
    # I have to modify D0 ~ D7 to binaryencoding
    data_x = df.loc[:, ['Timestamp', 'Arbitration_ID', 'DLC', 'Data']]
    data_y = df.loc[:, ['SubClass']]
