from cProfile import label
import torch
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class MLP(torch.nn.Module):
 
    def __init__(self, num_i, num_h, num_o):
        super(MLP,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.act1=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h)
        self.act2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_h)
        self.act3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(num_h,num_o)  
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.act3(x)
        x = self.linear4(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        x = data[:-1]
        #y_base = data[-2]
        y = data[-1]
        return x, y

def split(x, ratio=0.1):
    n_sample = x.shape[0]
    n_train = round(n_sample*(1-ratio))
    np.random.shuffle(x)
    train_sample = x[:n_train]
    test_sample = x[n_train:]
    return train_sample, test_sample

def train(mlp, train_loader, optimizer):
    losses = []
    for x, y in iter(train_loader):
        y_pred = mlp(x).reshape(-1)
        loss = torch.mean((y-y_pred)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.tolist())
    return losses

def test(mlp, test_loader):
    se = 0
    for x, y in iter(test_loader):
        y_pred = mlp(x).reshape(-1)
        se += torch.sum((y-y_pred)**2)
    mse = se/len(test_loader.dataset)
    mse = float(mse)
    print("MSE = {}".format(mse))
    return mse

def normalize(x):
    mu = np.mean(x, axis=0)
    std = np.clip(np.std(x, axis=0), 1e-5, 1e5)
    return (x-mu)/std        

def one_hot(x, feat):
    feat_expand = []
    x_expand = [[] for _ in range(x.shape[0])]
    for j in range(x.shape[1]):
        clss = list(set(x[:, j].astype(int)))
        for cls in clss:
            feat_expand.append(feat[j] + '-{}'.format(cls))
        for i in range(x.shape[0]):
            onehot = np.zeros(len(clss))
            onehot[clss.index(int(x[i][j]))] = 1
            x_expand[i].append(onehot)
    for i in range(x.shape[0]):
        x_expand[i] = np.concatenate(x_expand[i], axis=0)
    return np.array(x_expand), feat_expand

def MIV(mlp, train_loader, feat_name, delta=0.1):
    m = train_loader.dataset.data.shape[1] - 1
    miv = torch.zeros(m)
    for i in range(m):
        yp, yn = 0, 0
        for x, y in iter(train_loader):
            xp, xn = copy.deepcopy(x), copy.deepcopy(x)
            xp[:, i] *= 1 + delta
            xn[:, i] *= 1 - delta
            yp += mlp(xp).reshape(-1)
            yn += mlp(xn).reshape(-1)
        miv[i] = torch.mean(yp-yn)
    miv = torch.abs(miv)
    miv = miv / torch.sum(miv)
    miv = dict(zip(feat_name, miv.tolist()))
    miv = sorted(miv.items(), key=lambda x: x[1], reverse=True)
    return miv

def plot(train_loss, test_loss):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_loss, 'r', label='train')
    plt.plot(test_loss, 'b', label='test')
    plt.legend()
    plt.show()
    plt.savefig('loss.jpg')

def main():
    path = 'data.xlsx'
    data = pd.read_excel(path)
    flag = [int('.f' in col_name) for col_name in data.columns][:-1]
    is_discrete_feat = np.argwhere(np.array(flag) > 0).reshape(-1)
    is_continous_feat = np.argwhere(np.array(flag) == 0).reshape(-1)
    continous_feat = list(data.columns[is_continous_feat])
    discrete_feat = list(data.columns[is_discrete_feat])
    data = data.to_numpy().astype(np.float32)
    x = data[:, :-1]
    #y_base = data[:, -2].reshape(-1,1)
    y = data[:, -1].reshape(-1,1)
    x_con = x[:, is_continous_feat]
    x_dis = x[:, is_discrete_feat]

    x_dis, discrete_feat = one_hot(x_dis, discrete_feat)
    x = np.concatenate([x_con, x_dis], axis=1)
    feat_name = continous_feat + discrete_feat
    
    x = normalize(x)
    data = np.concatenate([x,y], axis=1).astype(np.float32)
    
    train_sample, test_sample = split(data, ratio=0.1)
    train_sample = torch.tensor(train_sample)
    test_sample = torch.tensor(test_sample)
    
    train_dataset = MyDataset(train_sample)
    train_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=True)
    
    test_dataset = MyDataset(test_sample)
    test_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=True)
    
    mlp = MLP(data.shape[1]-1, 10, 1)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    train_loss = []
    test_loss = []
    for e in range(1000):
        loss = train(mlp, train_loader, optimizer)
        train_loss.extend(loss)
        if e % 1 == 0:
            print('*'*10 + 'test at epoch {}'.format(e) + '*'*10)
            loss = test(mlp, test_loader)
            test_loss.append(loss)
            
    plot(train_loss, test_loss)
    print(MIV(mlp, train_loader, feat_name))
    
if __name__ == "__main__":
    main()