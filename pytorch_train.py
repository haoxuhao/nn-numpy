#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random

if __name__ == "__main__":
    random_seed=666
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # # iris dataset
    # iris = load_iris()
    # X = iris.data
    # target = iris.target
    # num_classes = 3
    # num_features = 4
    # # build a neural network
    # net = nn.Sequential(
    #     nn.Linear(num_features, 20),
    #     nn.ReLU(),
    #     nn.Linear(20, 15),
    #     nn.ReLU(),
    #     nn.Linear(15, num_classes),
    # )

    # mnist dataset
    mnist = load_digits()
    X = mnist.data
    target = mnist.target
    num_classes = 10
    num_features = 64
    # build a neural network
    net = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )

    X_normalized = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, \
        target.reshape(-1), test_size=0.30, random_state=random_seed)

    x_train, x_test = torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    
    # evaluate before train
    y_pred = net(x_test)
    y_pred = torch.argmax(y_pred, dim=1)
    acc_before_train = accuracy_score(y_test.numpy(), y_pred.numpy())

    # loss
    criterion = nn.CrossEntropyLoss()
    
    base_lr = 0.1
    optimizer = optim.SGD(net.parameters(), lr=base_lr)
    batch = 64
    epochs = 30

    # optimizer = optim.Adam(net.parameters(), lr=0.05)

    data_indexs = [a for a in range(X_train.shape[0])]

    for epoch in range(epochs):
        random.shuffle(data_indexs) #SGD shuffle training data indexs every epoch
        for i in range(0, len(data_indexs), batch):
            random_batch = data_indexs[i:i+batch]

            x = x_train[random_batch, :].reshape(-1, num_features)
            y_true = y_train[random_batch]

            # x = x_train[i:i+batch, :].reshape(-1, num_features)
            # y_true = y_train[i:i+batch]

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
    


    