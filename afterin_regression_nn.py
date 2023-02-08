import torch
import pandas as pd
import numpy as np
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
        x = data[:-2]
        y_base = data[-2]
        y = data[-1]
        return x, y_base, y

def split(x, ratio=0.1):
    n_sample = x.shape[0]
    n_train = round(n_sample*(1-ratio))
    np.random.shuffle(x)
    train_sample = x[:n_train]
    test_sample = x[n_train:]
    return train_sample, test_sample

def train(mlp, train_loader, optimizer):
    for x, y_base, y in iter(train_loader):
        y_pred = mlp(x).reshape(-1) + y_base
        loss = torch.mean((y-y_pred)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss)

def test(mlp, test_loader):
    se = 0
    for x, y_base, y in iter(test_loader):
        y_pred = mlp(x).reshape(-1) + y_base
        se += torch.sum((y-y_pred)**2)
    mse = se/len(test_loader.dataset)
    print("MSE = {}".format(float(mse)))

def normalize(x):
    mu = np.mean(x, axis=0)
    std = np.clip(np.std(x, axis=0), 1e-5, 1e5)
    return (x-mu)/std        

def one_hot(x, max_cls=10):
    one_hot_x = []
    for i in range(x.shape[0]):
        one_hot_x.append(np.eye(max_cls)[x[i].astype(int)].flatten())
    return np.array(one_hot_x)

def main():
    path = 'data.xlsx'
    data = pd.read_excel(path)
    flag = [int('.f' in col_name) for col_name in data.columns][:-2]
    is_discrete_feat = np.argwhere(np.array(flag) > 0).reshape(-1)
    is_continous_feat = np.argwhere(np.array(flag) == 0).reshape(-1)
    data = data.to_numpy().astype(np.float32)
    x = data[:, :-2]
    y_base = data[:, -2].reshape(-1,1)
    y = data[:, -1].reshape(-1,1)
    x_con = x[:, is_continous_feat]
    x_dis = x[:, is_discrete_feat]
    x_dis = one_hot(x_dis)
    x = np.concatenate([x_con, x_dis], axis=1)
    
    x = normalize(x)
    data = np.concatenate([x,y_base,y], axis=1).astype(np.float32)
    
    train_sample, test_sample = split(data, ratio=0.1)
    train_sample = torch.tensor(train_sample)
    test_sample = torch.tensor(test_sample)
    
    train_dataset = MyDataset(train_sample)
    train_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=True)
    
    test_dataset = MyDataset(test_sample)
    test_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=True)
    
    mlp = MLP(data.shape[1]-2, 10, 1)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    for e in range(1000):
        train(mlp, train_loader, optimizer)
        if e % 10 == 0:
            print('*'*10 + 'test at epoch {}'.format(e) + '*'*10)
            test(mlp, test_loader)
    
if __name__ == "__main__":
    main()