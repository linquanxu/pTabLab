# -*- coding: utf-8 -*-
from pandas import read_excel
from torch.utils.data import Dataset

class ExcelDataset(Dataset):
    def __init__(self, path):
        df = read_excel(path, header=None)
        self.x = df.values[:, :-1]
        self.y = df.values[:, -1]
        self.x = self.x.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]
    
    def __len__(self):
        return len(self.x)