#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import torch

# print(torch.optim.__file__)
import torch.nn as nn
import torch.optim as optim
import sys

if __name__ == "__main__":
    random_seed=666
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # prepare the dataset
    iris = load_iris()
    X = iris.data
    target = iris.target

    num_classes = 3
    num_features = 4
    X_normalized = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, \
        target.reshape(-1), test_size=0.30, random_state=random_seed)

    x_train, x_test = torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    
    net = nn.Sequential(
        nn.Linear(4, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 3),
    )

    # evaluate before train
    y_pred = net(x_test)
    y_pred = torch.argmax(y_pred, dim=1)
    acc_before_train = accuracy_score(y_test.numpy(), y_pred.numpy())

    # loss
    criterion = nn.CrossEntropyLoss()

    base_lr = 0.01
    # optimizer = optim.SGD(net.parameters(), lr=base_lr)
    batch = 16
    epochs = 150

    optimizer = optim.Adam(net.parameters(), lr=0.05)
    
    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch):
            x = x_train[i:i+batch, :].reshape(-1, num_features)
            y_true = y_train[i:i+batch]

            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print("epoch %d, batch %d, loss %.6f"%(epoch, i, loss.item()))

    y_pred = net(x_test)
    y_pred = torch.argmax(y_pred, dim=1)

    print("accuracy before train: %.3f"%acc_before_train)
    print("accuracy after train: %.3f"%accuracy_score(y_test.numpy(), y_pred.numpy()))
    


    