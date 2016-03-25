#!/usr/bin/python


import numpy as np
import sys
import argparse


class Layer:
    def __init__(self, lr=0.1):
        self.outputs = None
        self.lr = lr

    def forward(self, inputs):
        pass

    def backward(self, errors):
        pass

#Class to be used for matrix mult
class Linear(Layer):
    def __init__(self, num_inputs, num_nodes, **kwargs):
        Layer.__init__(self, **kwargs)
        self.weights = np.random.standard_normal([num_inputs, num_nodes])*0.1
        self.bias = np.ones(num_nodes)
        self.inputs = None

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.bias
        self.inputs = inputs
        return self.outputs

    def backward(self, errors):
        batch_size = self.inputs.shape[0]
        diff_input = np.dot(errors, np.transpose(self.weights))

        diff_weight = np.dot(np.transpose(self.inputs), errors)/batch_size
        diff_bias = errors.sum(axis=0)/batch_size

        self.weights += self.lr * diff_weight
        self.bias += self.lr * diff_bias

        return diff_input

class Sigmoid(Layer):
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-1 * inputs))
        return self.outputs

    def backward(self, errors):
        return errors * self.outputs * (1.0 - self.outputs) 


class Accuracy(Layer):
    def forward(self,inputs, labels):
        #print "Inputs:", inputs
        outputs = np.argmax(inputs, axis=1)
        #print "Output:", outputs
        #print "Label:", labels
        correct = np.equal(outputs, labels)
        #print "Correct array:", correct
        #print "Num_correct:", correct.sum()
        return correct.sum()

class Loss(Layer):
    def forward(self, inputs, labels):
        #print labels
        #print inputs

        targets = np.zeros(inputs.shape)
        targets[np.arange(inputs.shape[0]), labels] = 1.0

        error = targets - inputs
        self.output = error
        return np.square(self.output).sum()

    def backward(self, errors=None):
        return self.output 


class Network:
    def __init__(self, num_inputs, num_nodes=[8],  num_outputs=10, lr=0.1):

        self.layers = []

        prev_nodes = num_inputs
        for n in num_nodes:
            self.layers.append(Linear(prev_nodes, n, lr=lr))
            self.layers.append(Sigmoid())
            prev_nodes = n

        self.layers.append(Linear(prev_nodes, num_outputs))
        self.layers.append(Sigmoid())
        self.loss = Loss()
        self.accuracy = Accuracy()


    def forward(self, inputs):

        inp = inputs
        for l in self.layers:
            inp = l.forward(inp)

        return inp

    def backward(self):
        diff = self.loss.backward()
        for l in reversed(self.layers):
            diff = l.backward(diff)

    def trainBatch(self, inputs, labels, train=True):
        output = self.forward(inputs)
        loss = self.loss.forward(output, labels)
        correct = self.accuracy.forward(output, labels)
    
        if train:
            self.backward()

        #raw_input()

        return loss, correct

    def runEpoch(self, inputs, labels, batch_size, train=True):
        loss = 0.0
        correct = 0.0
        iteration = 0
        for i in range(0, inputs.shape[0], batch_size):
            iteration += 1
            data = inputs[i:i+batch_size,:]
            lbls = labels[i:i+batch_size]

            l, c = self.trainBatch(data,lbls, train=train)
            loss += l
            correct += c
            #print "Loss at iter %d: %f" %(iteration, l)
            #raw_input()


        #print "correct: %f (%d,%d)" % (correct/iteration/batch_size, correct, inputs.shape[0])
        return loss/iteration/batch_size, correct/iteration/batch_size
        

#read in data with optional shuffling
def parseCSV(filename, shuffle=False):
    data = []
    with open(filename, "r") as f:
        for line in f:
            l = [i.strip() for i in line.split(",")]
            data.append(l)

    #Make a np array while skipping header
    data = np.array(data[1:], dtype=float)

    #Optional shuffling
    if shuffle:
        np.random.shuffle(data)

    #Separate class label from features
    return data[:,:-1], data[:,-1].astype(int)



def main(args):
    data, labels = parseCSV(args.data, shuffle=True)

    idx = round(data.shape[0]*args.split)

    x_train, y_train = data[:idx], labels[:idx]
    x_test, y_test = data[idx:], labels[idx:]

    outputs = len(set(labels))

    net = Network(num_inputs=data.shape[1], num_nodes=args.num_nodes, num_outputs=outputs, lr = args.lr)


    #print net.layers
    #print net.forward(data[0:2])

    for i in range(args.epochs):
        loss = net.runEpoch(x_train,y_train,args.batch_size)
        if i%10 == 0:
            print "Training Epoch Loss and Accuracy:", loss

        if i%30 == 0 or i == args.epochs - 1:
            loss = net.runEpoch(x_test, y_test,args.batch_size,train=False)
            print "Test Epoch Loss and Accuracy:", loss
            print



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network example.")
    parser.add_argument("data", help="The path to the csv data file.")
    parser.add_argument("-e", "--epochs", help="Number of epochs to run.", type=int, default=100) 
    parser.add_argument("-n", "--num_nodes", help="A list of the number of nodes in each hidden layer", nargs='+', type=int, default=["8"])
    parser.add_argument("-b", "--batch_size", help="The mini-batch size.", type=int, default=1)
    parser.add_argument("-lr", help="The learning rate", type=float, default=0.1)
    parser.add_argument("-s", "--split", help="The percentage of data to be used for training.", type=float, default=0.7)

    args = parser.parse_args()
    main(args)

