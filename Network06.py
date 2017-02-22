import numpy as np
import matplotlib.pyplot as plt
from random import random


def plotting():

    global pl

    fig = plt.figure()

    for i in range(5):
        pl.performTeaching()
    draw1 = fig.add_subplot(331)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw1.plot(i, j, color)
    draw1.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw1.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw1.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw2 = fig.add_subplot(332)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw2.plot(i, j, color)
    draw2.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw2.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw2.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw3 = fig.add_subplot(333)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw3.plot(i, j, color)
    draw3.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw3.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw3.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw4 = fig.add_subplot(334)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw4.plot(i, j, color)
    draw4.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw4.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw4.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw5 = fig.add_subplot(335)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw5.plot(i, j, color)
    draw5.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw5.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw5.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw6 = fig.add_subplot(336)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw6.plot(i, j, color)
    draw6.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw6.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw6.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw7 = fig.add_subplot(337)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw7.plot(i, j, color)
    draw7.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw7.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw7.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw8 = fig.add_subplot(338)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw8.plot(i, j, color)
    draw8.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw8.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw8.axis([-1, 1, -1, 1])

    for i in range(5):
        pl.performTeaching()
    draw9 = fig.add_subplot(339)
    for i in np.arange(-1, 1.08, 0.08):
        for j in np.arange(-1, 1.08, 0.08):
            if pl.test([i, j]) > 0:
                color = 'ro'
            else:
                color = 'bo'
            draw9.plot(i, j, color)
    draw9.plot([0.4, 0.4, 0.5, 0.5, 0.04], [0.4, 0.5, 0.4, 0.5, 0.04], 'ko')
    draw9.plot([-0.4, -0.4, -0.5, -0.5, -0.04], [-0.4, -0.5, -0.4, -0.5, -0.04], 'wo')
    draw9.axis([-1, 1, -1, 1])
    plt.show()


def createTeachingSet():
    teachingSet.listOfElements.append(Element([0.4, 0.4], -1, "Object 1, class 1"))
    teachingSet.listOfElements.append(Element([0.4, 0.5], -1, "Object 2, class 1"))
    teachingSet.listOfElements.append(Element([0.5, 0.4], -1, "Object 3, class 1"))
    teachingSet.listOfElements.append(Element([0.5, 0.5], -1, "Object 4, class 1"))
    teachingSet.listOfElements.append(Element([0.04, 0.04], -1, "Object 5, class 1"))
    teachingSet.listOfElements.append(Element([-0.04, -0.04], 1, "Object 1, class 2"))
    teachingSet.listOfElements.append(Element([-0.5, -0.5], 1, "Object 2, class 2"))
    teachingSet.listOfElements.append(Element([-0.4, -0.5], 1, "Object 3, class 2"))
    teachingSet.listOfElements.append(Element([-0.4, -0.4], 1, "Object 4, class 2"))
    teachingSet.listOfElements.append(Element([-0.5, -0.4], 1, "Object 5, class 2"))


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
        print(self.comment, end=" ")
        print("Inputs: ", self.inputs, end=" ")
        print("Output: ", self.expectedOutput)


class NeuralNetwork:
    """Artificial neural network, consist of neurons."""
    def __init__(self, inputCount):
        self.inputCount = inputCount
        self.neuron = Neuron(self.inputCount)

    def learn(self, teachingElement, ratio, previousResponse, previousError):
        self.neuron.learn(
            teachingElement.inputs,
            teachingElement.expectedOutput,
            ratio,
            previousResponse,
            previousError
            )


class Neuron:
    """Artificial neuron - every example has its own weights which change in every learning step."""

    def __init__(self, neuronInputs):
        self.weights = []
        self.randomize(-1, 1, neuronInputs)

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
    """Main logic of the network."""

    def __init__(self, teachingSet, network):
        self.teachingElementIndex = 0
        self.ratio = 0.1
        self.teachingSet = teachingSet
        self.teachingElement = Element
        self.examinedNetwork = network
        self.previousResponse = [0.]
        self.previousError = [0.]

    def performTeaching(self):
        """Conduct teaching network.
            Choose randomly element from the teaching set and teach network how to behave.
        """
        self.teachingElementIndex = int(random() * self.teachingSet.count())
        self.teachingElement = self.teachingSet.listOfElements[self.teachingElementIndex]
        self.examinedNetwork.learn(
            self.teachingElement,
            self.ratio,
            self.previousResponse,
            self.previousError
            )

    def test(self, inputs):
        return self.examinedNetwork.neuron.response(inputs)

print("Neural network with 1 non-linear neuron")
print("Teaching Set:")
teachingSet = TeachingSet(2, 1)
createTeachingSet()
teachingSet.show()

network = NeuralNetwork(2)
pl = ProgramLogic(teachingSet, network)

print("Every next picture shows network response after 5 teaching steps.")
for k in range(10):
    plotting()
    _ = input('If you want to show next pictures, close plot and enter any character, if you want to exit, enter q')
    if _ == 'q':
        exit()






