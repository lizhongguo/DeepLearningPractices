import numpy as np
import random
import math
import pdb
from tqdm import *

# Full Connected Layer
class FC:
    def __init__(self,size,lr=0.001):
        assert(len(size)==2)
        self.lr = lr
        self.size = size
        self.weights = np.matrix(np.random.rand(size[1],size[0]))
        self.bias = np.matrix(np.random.rand(size[1],1))
        self.i_cache = np.matrix(np.random.rand(size[0],1))

    def forward(self,i):
        self.i_cache = i
        return np.matmul(self.weights, i) + self.bias

    def backward(self,d_o):
        d_bias = d_o
        d_weights = np.matmul(d_o, self.i_cache.transpose())
        d_i = np.matmul(self.weights.T, d_o)

        self.weights = self.weights - self.lr*d_weights
        self.bias = self.bias - self.lr*d_bias

        return d_i

# Loss Function
class L2:
    def forward(self,output,label):
        diff = output - label
        return np.dot(diff,diff)

    def backward(self,output,label):
        return 2*(output-label)

# Activation Function
class ReLU:

    def forward(i):
        return i

    def backward(o):
        return o


# Neural Networks
class NN:
    def __init__(self,size,lr=0.001):
        self.size = size
        self.lr = lr
        assert(len(size)>=3)
        self.layers = []

        for i in range(len(size)-1):
            self.layers.append(FC(size[i:i+2],lr))

    def forward(self,i):
        for layer in self.layers:
            i = layer.forward(i)
        return i

    def backward(self,d_o):
        for layer in reversed(self.layers):
            d_o = layer.backward(d_o)
        return d_o


def train(Net, LF, dataset, epoch = 10000):
    all_epoch = len(dataset)
    for i in range(20):
        epoch = 0
        loss = 0
        for batch in tqdm(dataset):
            data = batch[0]
            label = batch[1]
            output = Net.forward(data)
            loss = loss + LF.forward(output=output, label=label)
            epoch = epoch + 1
            #print("Loss {}  {}/{} epoch".format(loss, epoch, all_epoch))
            d_o = LF.backward(output=output, label=label)
            Net.backward(d_o)
        print("Loss {}".format(loss))

def generate_dataset(size = 10000):
    dataset = []
    for i in range(size):
        d = np.matrix([random.random(),])
        l = np.matrix([math.sin(d),])
        dataset.append((d,l))
    return dataset

if __name__ == '__main__':
    dataset = generate_dataset()
    Net = NN(size=(1,100,10,1),lr=0.00001)
    LF = L2()
    train(Net, LF, dataset, epoch = 10000)

