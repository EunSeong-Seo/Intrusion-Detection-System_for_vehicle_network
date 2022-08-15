import pandas as pd


def make_train_xy_df(path):
    # Load data from path
    df = pd.read_csv(path)
    features = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'Class', 'SubClass']
    data_x = df.loc[:, ['Timestamp', 'Arbitration_ID']]
    data_y = df.loc[:, ['SubClass']]
    data_y = 


def make_train_xy_df_with_D(path):
    # Load data from path
    df = pd.read_csv(path)
    features = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'Class', 'SubClass']
    df_data = df['Data'].str.split(' ', expand=True)
    df_data.columns = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
    df_data
    # I have to modify D0 ~ D7 to binaryencoding
    data_x = df.loc[:, ['Timestamp', 'Arbitration_ID', 'DLC', 'Data']]
    data_y = df.loc[:, ['SubClass']]