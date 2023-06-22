# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import torch
from torch.utils.data import DataLoader
from numpy import vstack
from sklearn.metrics import accuracy_score
from torch.nn import BCELoss
from torch import optim
from models.dnn import DNN
from datasets.Excel import ExcelDataset

def eval_model(test_dataloader, model):
    predictions, labels = [], []
    # savefile = open('save_model\\results_anbili_new.txt','w')
    for i, (inputs, label) in enumerate(test_dataloader):
        pred = model(inputs)
        pred = pred.detach().numpy()
        label = label.numpy()
        label = label.reshape((len(label), 1))
        # savefile.write(str(label[0][0])+' '+str(x[0][0])+'\n')
        pred = pred.round()
        predictions.append(pred)
        labels.append(label)
    predictions, labels = vstack(predictions), vstack(labels)
    acc = accuracy_score(labels, predictions)
    return acc

def train_model(train_path,test_path, model):
    train_dataset = ExcelDataset(train_path)
    test_dataset = ExcelDataset(test_path)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    criterion = BCELoss()
    #criterion= FocalLoss()
    betas = (0.9, 0.999)
    lr = 0.0001
    weight_decay = 0.5e-5
    eps = 1e-9
    #optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
    for epoch in range(400):
        a_loss=0
        for i, (inputs, tg) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = model(inputs)
            loss = criterion(x, tg)
            loss.backward()
            optimizer.step()
            a_loss = loss.data
        acc = eval_model(val_dataloader, model)
        print("epoch: {}, acc: {}, loss: {}".format(epoch, acc, a_loss))
        torch.save(model.state_dict(),'work_dirs/epoch_'+str(epoch)+'.pth')

if __name__ == "__main__":
    train_path = 'train.xls'
    test_path = 'test.xls'
    model_path = 'work_dirs\\epoch_119.pth'
    test_dataset = ExcelDataset(test_path)
    val_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model = DNN(25)
    # train_model(train_path, test_path, model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    acc = eval_model(val_dataloader, model)
    print('Accuracy: %.3f' % acc)