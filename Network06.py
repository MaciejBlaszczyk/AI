import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from random import random

#x = np.array([1,2,3,4,10])


#plt.plot(x, '.')
#plt.show()


def createTeachingSet():
    teachingSet.listOfElements.append(Element([0.1, 0.4], -1, "Object 1, class 1"))
    teachingSet.listOfElements.append(Element([0.6, 0.5], -1, "Object 2, class 1"))
    teachingSet.listOfElements.append(Element([0.4, 0.4], -1, "Object 3, class 1"))
    teachingSet.listOfElements.append(Element([0.5, 0.5], -1, "Object 4, class 1"))
    teachingSet.listOfElements.append(Element([0.2, 0.7], -1, "Object 5, class 1"))
    teachingSet.listOfElements.append(Element([-0.6, -0.2], 1, "Object 1, class 2"))
    teachingSet.listOfElements.append(Element([-0.5, -0.5], 1, "Object 2, class 2"))
    teachingSet.listOfElements.append(Element([-0.8, -0.5], 1, "Object 3, class 2"))
    teachingSet.listOfElements.append(Element([-0.7, -0.3], 1, "Object 4, class 2"))


class TeachingSet:
    """Teaching set, consist of elements list, number of inputs and outputs in every element."""

    def __init__(self, inputCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.listOfElements = []

    def show(self):
        for element in self.listOfElements:
            element.show()

    def count(self):
        return len(self.listOfElements)


class Element:
    """Single element of the teaching set, consists of: list of inputs, expected output, comment."""

    def __init__(self, inputs, expectedOutput, comment):
        self.inputs = inputs
        self.expectedOutput = expectedOutput
        self.comment = comment

    def show(self):
        print(self.comment)
        print("Inputs: ", self.inputs)
        print("Output: ", self.expectedOutput)


class NeuralNetwork:
    def __init__(self, inputCount):
        self.biasInput = 1
        self.inputCount = inputCount
        self.neuronInputs = self.inputCount + self.biasInput
        self.neuron = Neuron(self.neuronInputs)

    def appendBias(self, signals):
        result = list(signals)
        result.append(1)
        return result

    def learn(self, teachingElement, ratio, previousResponse, previousError):
        actualInputs = self.appendBias(teachingElement.inputs)
        self.neuron.learn(
            actualInputs,
            teachingElement.expectedOutput,
            ratio,
            previousResponse,
            previousError
            )


class Neuron:
    """Artificial neuron - every example has its own weights which change in every learning step."""

    def __init__(self, neuronInputs):
        self.weights = []
        self.randomize(-0.1, 0.1, neuronInputs)

    def randomize(self, min, max, inputCount):
        """Set neuron weights to random numbers between min and max."""
        length = max - min
        for i in range(inputCount):
            self.weights.append(min + length * random())

    def learn(self, signals, expectedOutputs, ratio, previousResponse, previousError):
        """Set neuron weights depending on ratio, previous error and input signals."""
        previousResponse[0] = self.response(signals)
        previousError[0] = expectedOutputs - previousResponse[0]
        for i in range(len(self.weights)):
            self.weights[i] += ratio * previousError[0] * signals[i]

    def response(self, signals):
        """Return neuron's response (sum of multiplied weights and input signals)."""
        result = 0.
        for signal, weight in zip(signals, self.weights):
            result += signal * weight
        return result


class ProgramLogic:

    def __init__(self, teachingSet, network):
        self.teachingElementIndex = 0
        self.ratio = 0.1
        self.teachingSet = teachingSet
        self.teachingElement = Element
        self.examinedNetwork = network
        self.previousResponse = [0.]
        self.previousError = [0.]

    def performTeaching(self):
        self.teachingElementIndex = int(random() * self.teachingSet.count())
        self.teachingElement = self.teachingSet.listOfElements[self.teachingElementIndex]
        self.examinedNetwork.learn(
            self.teachingElement,
            self.ratio,
            self.previousResponse,
            self.previousError
            )



teachingSet = TeachingSet(2, 1)
createTeachingSet()
teachingSet.show()

network = NeuralNetwork(2)
pl = ProgramLogic(teachingSet, network)
for i in range(1000):
    pl.performTeaching()



print(teachingSet.listOfElements[0].inputs)
plt.plot(teachingSet.listOfElements[0].inputs[0], teachingSet.listOfElements[0].inputs[1], 'ro')
plt.plot(teachingSet.listOfElements[1].inputs[0], teachingSet.listOfElements[1].inputs[1], 'ro')
plt.plot(teachingSet.listOfElements[2].inputs[0], teachingSet.listOfElements[2].inputs[1], 'ro')
plt.plot(teachingSet.listOfElements[3].inputs[0], teachingSet.listOfElements[3].inputs[1], 'ro')
plt.plot(teachingSet.listOfElements[4].inputs[0], teachingSet.listOfElements[4].inputs[1], 'ro')
plt.plot(teachingSet.listOfElements[5].inputs[0], teachingSet.listOfElements[5].inputs[1], 'bs')
plt.plot(teachingSet.listOfElements[6].inputs[0], teachingSet.listOfElements[6].inputs[1], 'bs')
plt.plot(teachingSet.listOfElements[7].inputs[0], teachingSet.listOfElements[7].inputs[1], 'bs')
plt.plot(teachingSet.listOfElements[8].inputs[0], teachingSet.listOfElements[8].inputs[1], 'bs')
plt.plot(0.1,0.1,'ro')
plt.plot(0.12,0.12,'ro')
plt.axis([-1, 1, -1, 1])


plt.show()