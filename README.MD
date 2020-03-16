# Neural Network in pure Numpy
Implemented a fully connected neural network in pure NumPy and compared the performance with Pytorch on the Iris dataset.

# Run
``` python
# train and get the test accuracy 
python train.py

'''
output
...
epoch 149, batch 80, loss 0.101833
epoch 149, batch 96, loss 0.221098
accuracy before train: 0.400
accuracy after train: 0.978
'''

# train and get the test accuracy using PyTorch
python pytorch_train.py

'''
output
...
epoch 149, batch 80, loss 0.000268
epoch 149, batch 96, loss 0.000433
accuracy before train: 0.333
accuracy after train: 0.956
'''
```