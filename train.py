import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.nn import Sequential
import torch.optim as optim
from random import shuffle


train_data = pd.read_csv("data/fall_vs_up.csv",header=None)
# print(train_data[5])
print(train_data)

def get_data(file):
    inputs = []
    labels = []
    with open(file) as file:
        df = pd.read_csv(file, header=None)
        x = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        for i in range(len(x)):
            inputs.append(x[i])
        for j in range(len(y)):
            a = []
            if(y[j]=="fall"):
                y[j]=1
            else:
                y[j]=0
            a.append(y[j])
            labels.append(a)
    return inputs,labels
inputs,outputs=get_data("data/fall_vs_up.csv")
print(inputs[1])
print(outputs[1])


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = Sequential(
            nn.Linear(99, 10),
            nn.ReLU(),
            nn.Linear(10,2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 模型定义和训练
shwj=MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(shwj.parameters(), lr=1e-4)
# optimizer = optim.SGD(shwj.parameters(), lr=1e-4, momentum=0.9)
EPOCH = 50

for epoch in range(EPOCH):
    step = 0
    loss_one = 0
    list_shuffle = [i for i in range(len(inputs))]  # 某个排列
    shuffle(list_shuffle)
    for i in list_shuffle:
        train_loss = 0
        x = inputs[i]
        y = outputs[i]
        y = torch.tensor(y)
        x = torch.tensor(x).unsqueeze(0).to(torch.float32)
        output = shwj(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(inputs)

    print('Epoch: {} \t Training Loss:{:.6f}'.format(epoch + 1, train_loss))

torch.save(shwj,"checkpoint/fall_vs_up.pth")