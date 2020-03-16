from functions import *
import math

class CrossEntropyLoss:
    def __init__(self, reduce="mean"):
        self.reduce = reduce
        self.softmax_y = None
        self.y_true = None
    
    def __call__(self, y, y_true):
        loss = self.cross_entropy(y, y_true)
        self.grad = self.cross_entropy_grad()
        return loss
            
    def cross_entropy(self, y, y_true):
        self.softmax_y = softmax(y) #softmax before calculate cross entropy
        self.y_true = y_true # save for calculating grad
        return cross_entropy(self.softmax_y, self.y_true, reduce=self.reduce)
    
    def cross_entropy_grad(self):
        grad = -(self.y_true - self.softmax_y) #y - y_true
        return grad

class Layer:
    def __init__(self, in_dim, out_dim, activation_func="relu", init_var = 0.01):
        # self.w = np.random.rand(out_dim, in_dim)*math.sqrt(init_var)
        # self.b = np.zeros((out_dim, 1))

        sqrt_k = math.sqrt(1/in_dim)
        self.w = np.random.uniform(-sqrt_k, sqrt_k, (out_dim, in_dim))
        self.b = np.random.uniform(-sqrt_k, sqrt_k, (out_dim, 1))

        assert activation_func in ["relu", "sigmoid", "linear"]
        
        self.activation_func = eval(activation_func)
        self.activation_func_grad = eval(activation_func+"_grad")
        
        self.in_cache = None
        self.z_cache = None #value before activation function
        self.grad_w = None
        self.grad_b = None
        
    def forward(self, x):
        self.in_cache = x
        out = np.dot(self.w, x.T) + self.b
        self.z_cache = out.T #axis 0 is num of samples
        out = self.activation_func(self.z_cache)
        return out
    
    def no_grad(self):
        '''
        clear cache
        '''
        self.in_cache = None
        self.z_cache = None
        self.grad_w = None
        self.grad_b = None

    def backward(self, grad):
        '''
        chain rule
        '''
        activation_grad = self.activation_func_grad(self.z_cache)
        dz = activation_grad*grad
        nsmaples = dz.shape[0]

        self.grad_w = np.dot(dz.T, self.in_cache)/nsmaples
        self.grad_b = np.mean(dz, axis=0, keepdims=True).T

        # this is the most important part of backward, reflects the basic idea of BP
        # the gradient transfer to the former layer through the inverse connection of neural network
        # summation of the influence from Loss function to the hidden unit
        ret_grad = np.dot(dz, self.w) 
        
        return ret_grad
    
    def update(self, lr):
        self.w -= lr*self.grad_w
        self.b -= lr*self.grad_b

class NN:
    def __init__(self, layers=[], activations=[]):
        self.nn = []
        assert len(layers)-1 == len(activations)

        for i in range(len(layers)-1):
            in_dim = layers[i]
            out_dim = layers[i+1]
            self.nn.append(Layer(in_dim, out_dim, activations[i]))

    def forward(self, x):
        for l in self.nn:
            x = l.forward(x)
        
        return x
        
    def backward(self, grad):
        for l in reversed(self.nn):
            grad = l.backward(grad)
            
        # return the grad to the input data
        return grad
    
    def update(self, lr):
        '''
        lr is the learning rate
        '''
        for l in self.nn:
            l.update(lr)

    def no_grad(self):
        for l in self.nn: l.no_grad()
