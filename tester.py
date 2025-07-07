import numpy as np


# testData = np.loadtxt("test.csv", delimiter=",", skiprows=1)
testData = np.loadtxt("train.csv", delimiter=",", skiprows=1)

model = np.load("trained_model_100epochs.npz")

randomWeightsL1 = model["W1"]
randomBiasL1 = model["B1"]
randomWeightsL2 = model["W2"]
randomBiasL2 = model["B2"]
randomWeightsL3 = model["W3"]
randomBiasL3 = model["B3"]


# testPixels = testData / 255
testPixels = testData[:, 1:] / 255
trainLabels = testData[:, 0].astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deri(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def makeAnsArray(labels):
    one_hot = np.zeros((labels.size, 10))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def costFuncRes(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred), axis=1)


input_size = 784
hidden1_size = 32
hidden2_size = 16
output_size = 10


# between inp lay and 32 neuron lay


z1 = testPixels @ randomWeightsL1 + randomBiasL1
a1 = sigmoid(z1)
layer32neuron =  sigmoid(testPixels @ randomWeightsL1 + randomBiasL1)

# bw 32 lay and 16 lay
z2 = a1 @ randomWeightsL2 + randomBiasL2
a2 = sigmoid(z2)
layer16neuron = sigmoid(layer32neuron @ randomWeightsL2  + randomBiasL2)

# between 16 lay and out lay

z3 = a2 @ randomWeightsL3 + randomBiasL3
a3 = softmax(z3)
outputNeuron = softmax(layer16neuron @ randomWeightsL3 + randomBiasL3)




predictedArray = np.argmax(outputNeuron, axis=1)


mistakeMask = predictedArray != trainLabels

print(trainLabels[mistakeMask])