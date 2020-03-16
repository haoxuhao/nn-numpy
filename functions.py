import numpy as np

def sigmoid(x): 
    return 1/(1+np.exp(-x))

def relu(x):
    x[np.where(x<0)] = 0
    return x

def linear(x):
    return x

def softmax(x):
    '''Compute the softmax in a numerically stable way.'''
    x = x - np.max(x)
    x_exp = np.exp(x)
    sum_exp = np.sum(x_exp, axis=1).reshape(x.shape[0], -1)
    return x_exp/sum_exp

def softmax_grad(x):
    '''
    it's better to combine the cross entropy with softmax to calculate the gradient
    '''
    pass
def relu_grad(x):
    grad = np.where(x < 0, 0, 1)
    return grad
    
def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))

def linear_grad(x):
    return 1
    
def cross_entropy(y, y_true, reduce="mean"):
    if reduce=="mean":
        return np.sum(-y_true*np.log(y+1e-12))/y.shape[0]
    else:
        return np.sum(-y_true*np.log(y+1e-12))
    
def cross_entropy_grad(y, y_true):
    '''
    it's better to combine the cross entropy with softmax to calculate the gradient
    '''
    pass

def binary_cross_entropy_grad(y, y_true):
    '''
    y.shape = y_true.shape: [m, 1] 1 is the output dim, m is the nsmaples 
    '''
    return -(y_true*(1/(y+1e-12)) + (1-y_true)*1/(1-y+1e-12))

def binary_cross_entropy(y, y_true, reduce="mean"):
    '''
    y.shape = y_true.shape: [m, 1] 1 is the output dim, m is the nsmaples 
    L = -(yilogy+(1-y)log(1-y))
    
    '''
    if reduce=="mean":
        return np.mean(-(y_true*np.log(y+1e-12) + (1-y_true)*np.log(1-y+1e-12)))
    else:
        return np.sum(-(y_true*np.log(y+1e-12) + (1-y_true)*np.log(1-y+1e-12)))
    
