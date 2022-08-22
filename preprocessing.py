import pandas as pd

def change_hex_to_int(x):
    if x is None:
        return int(-1)
    return int(x, 16)


def make_train_xy_df(path):
    # Load data from path
    df = pd.read_csv(path)
    features = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'Class', 'SubClass']

    # Change hex to int and fill -1 at NaN data
    df['Arbitration_ID'] = df['Arbitration_ID'].apply(change_hex_to_int)
    data_x = df.loc[:, ['Timestamp', 'Arbitration_ID']]
    data_y = df.loc[:, ['SubClass']]
    return data_x, data_y


def make_data_target(path):
    # Load data from path
    df = pd.read_csv(path)
    features = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data', 'Class', 'SubClass']
    df_data = df['Data'].str.split(' ', expand=True)
    df_data.columns = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

    # concat Arbitraion ID, D0 ~ D7 columns after Changing hex to int or fill -1 at NaN data
    df['Arbitration_ID'] = df['Arbitration_ID'].apply(change_hex_to_int)
    df_data['D0'] = df_data['D0'].apply(change_hex_to_int)
    df_data['D1'] = df_data['D1'].apply(change_hex_to_int)
    df_data['D2'] = df_data['D2'].apply(change_hex_to_int)
    df_data['D3'] = df_data['D3'].apply(change_hex_to_int)
    df_data['D4'] = df_data['D4'].apply(change_hex_to_int)
    df_data['D5'] = df_data['D5'].apply(change_hex_to_int)
    df_data['D6'] = df_data['D6'].apply(change_hex_to_int)
    df_data['D7'] = df_data['D7'].apply(change_hex_to_int)
    data_x = pd.concat([df.loc[:, ['Timestamp', 'Arbitration_ID', 'DLC']], df_data], axis=1)
    data_y = df.loc[:, ['SubClass']]
    return data_x, data_y


if __name__ == '__main__':
    import pandas as pd
    path = 'dataset/Pre_train_D_attack_1.csv'
