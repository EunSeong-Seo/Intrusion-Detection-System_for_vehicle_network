import random
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from preprocessing import make_data_target

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCH = 400
LR = 0.001
BS = 16384
SEED = 41


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Seed 고정
seed_everything(SEED)

# Load dataset
train = pd.read_csv('dataset/Pre_train_D_attack_free.csv')
train = make_data_target(train)
val = pd.read_csv('dataset/Pre_train_D_attack_1.csv')
val = make_data_target(val)
test = pd.read_csv('dataset/Pre_train_D_attack_2.csv')
tset = make_data_target(test)


class MyDataset(Dataset):
    def __init__(self, df, eval_mode):
        self.df = df
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.labels = self.df['Class'].values
            self.df = self.df.drop(columns=['Class']).values
        else:
            self.df = self.df.values

    def __getitem__(self, index):
        if self.eval_mode:
            x = torch.from_numpy(self.df[index]).type(torch.FloatTensor)
            y = torch.FloatTensor([self.labels[index]])
            return x, y
            # self.x = self.df[index]
            # self.y = self.labels[index]
            # return torch.Tensor(self.x), self.y
        else:
            self.x = self.df[index]
            return torch.Tensor(self.x)

    def __len__(self):
        return len(self.df)


train_dataset = MyDataset(df=train, eval_mode=False)
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=6)

val_dataset = MyDataset(df = val, eval_mode=True)
val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=6)