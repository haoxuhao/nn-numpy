#!/usr/local/bin/python3

from feedforward_nn import NN, CrossEntropyLoss
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from functions import softmax
from sklearn.metrics import accuracy_score
import math


if __name__ == "__main__":
    random_seed=666
    np.random.seed(random_seed)

    # prepare the dataset
    iris = load_iris()
    X = iris.data
    target = iris.target

    num_classes = 3
    num_features = 4
    X_normalized = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, \
        target.reshape(-1), test_size=0.30, random_state=random_seed)

    y_train_onehot = np.eye(num_classes)[y_train.reshape(-1)]

    # build a neural network
    net = NN(layers=[num_features, 20, 15, num_classes], activations=["relu", "relu", "linear"])
    
    # loss
    criteria = CrossEntropyLoss()

    # hyperparameters
    base_lr = 0.01
    batch = 16
    epochs = 150

    #eval before training
    test_output = net.forward(X_test)
    y_pred = np.argmax(test_output, axis=1)
    acc_before_train = accuracy_score(y_test, y_pred)
    
    #clear cache
    net.no_grad()

    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch):
            x = X_train[i:i+batch, :].reshape(-1, num_features)
            y_true = y_train_onehot[i:i+batch, :].reshape(-1, num_classes)

            y = net.forward(x)
            loss = criteria(y, y_true)
            net.backward(criteria.grad)

            lr = base_lr * 1 #math.pow(0.1, epoch//100)
            
            net.update(lr)

            print("epoch %d, batch %d, loss %.6f"%(epoch, i, loss))
            # break


    #eval after training
    test_output = net.forward(X_test)
    y_pred = np.argmax(test_output, axis=1)

    print("accuracy before train: %.3f"%acc_before_train)
    print("accuracy after train: %.3f"%accuracy_score(y_test, y_pred))
    


    
    




