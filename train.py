#!/usr/local/bin/python3

from feedforward_nn import NN, CrossEntropyLoss
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from functions import softmax
from sklearn.metrics import accuracy_score
import math
from tqdm import tqdm
import random

def train():
    random_seed=666
    np.random.seed(random_seed)

    # mnist dataset
    mnist = load_digits()
    X = mnist.data
    target = mnist.target
    num_classes = 10
    num_features = 64
    # build a neural network
    net = NN(layers=[num_features, 256, 128, 64, 32, num_classes], activations=["relu", "relu", "relu", "relu", "linear"])

    # # iris dataset
    # iris = load_iris()
    # X = iris.data
    # target = iris.target
    # num_classes = 3
    # num_features = 4
    # # build a neural network
    # net = NN(layers=[num_features, 20, 10, num_classes], activations=["relu", "relu", "linear"])
    
    # loss
    criteria = CrossEntropyLoss()

    # hyperparameters
    base_lr = 0.1
    batch = 64
    epochs = 30
    

    X_normalized = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, \
        target.reshape(-1), test_size=0.30, random_state=random_seed)

    print("train size: %d; test size: %d"%(X_train.shape[0], X_test.shape[0]))

    y_train_onehot = np.eye(num_classes)[y_train.reshape(-1)]

    #eval before training
    def eval(net, eval_data, eval_label):
        test_output = net.forward(eval_data)
        y_pred = np.argmax(test_output, axis=1)
        acc = accuracy_score(eval_label, y_pred)
        #clear cache
        net.no_grad()
        return acc

    acc_before_train = eval(net, X_test, y_test)
    acc = acc_before_train

    data_indexs = [a for a in range(X_train.shape[0])]
    lr = base_lr
    loss = 0
    for epoch in range(epochs):
        random.shuffle(data_indexs)
        for i in tqdm(range(0, len(data_indexs), batch), desc="epoch %d; train loss %.4f; test acc %.3f; lr %.5f"%(epoch+1, loss, acc, lr)):
            if i==0: loss=0
            indexs = data_indexs[i:i+batch]
            x = X_train[indexs, :].reshape(-1, num_features)
            y_true = y_train_onehot[indexs, :].reshape(-1, num_classes)

            y = net.forward(x)
            loss += criteria(y, y_true)
            net.backward(criteria.grad)

            lr = base_lr
            
            net.update(lr)

            # print("epoch %d, batch %d, loss %.6f"%(epoch, i, loss), flush=True)

        acc = eval(net, X_test, y_test)
        

    print("accuracy before train: %.3f"%acc_before_train)
    print("accuracy after train: %.3f"%acc)


if __name__ == "__main__":
    train()


    
    




