from math import sqrt
from random import random


def normalize(signals):
    strength = 0
    for signal in signals:
        strength += signal * signal
    strength = sqrt(strength)
    for i in range(len(signals)):
        signals[i] /= strength


class Neuron:
    weights = []
    prevWeights = []

    def __init__(self):
        pass

    def randomize(self, min, max, inputCount):
        length = max - min
        for i in range(inputCount):
            self.weights.append(min + length * random())

    def learn(self, signals, expectedOutput, ratio, previousResponse, previousError):
        previousResponse[0] = self.response(signals)
        previousError[0] = expectedOutput - previousResponse[0]
        for i in range(len(self.weights)):
            self.weights[i] += ratio * previousError[0] * signals[i]

    def response(self, signals):
        result = 0.
        for signal, weight in zip(signals, self.weights):
            result += signal * weight
        return result


class Element:
    inputs = []
    expectedOutput = 0.
    comment = ""

    def __init__(self, inputs, expectedOutput, comment):
        self.inputs = inputs
        self.expectedOutput = expectedOutput
        self.comment = comment

    def clone(self):
        elem = Element()
        elem.inputs = self.inputs
        elem.expectedOutput = self.expectedOutput
        return elem

    def show(self):
        print(self.inputs)
        print(self.expectedOutput)


class TeachingSet:
    listOfElements = []
    inputCount = 0.
    outputCount = 0.

    def __init__(self, inputCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount

    def normalize(self):
        for item in self.listOfElements:
            normalize(item.inputs)

    def show(self):
        for element in self.listOfElements:
            element.show()

    def readFromConsole(self):
        print("Enter number of inputs:")
        self.inputCount = int(input())
        print("Enter number of outputs:")
        self.outputCount = int(input())
        pass

    def count(self):
        return len(self.listOfElements)


class ProgramLogic:
    currentElementIndex = 0
    currentNormalizedInputs = []
    previousWeights = []
    previousResponse = [0.]
    previousError = [0.]
    currentResponse = 0.
    currentError = 0.
    examinedNeuron = Neuron
    teachingSet = TeachingSet
    currentElement = Element
    history = []

    def __init__(self, teachingSet, neuron):
        self.teachingSet = teachingSet
        self.examinedNeuron = neuron

    def performTeaching(self, teachingRatio):

        if self.currentElementIndex >= self.teachingSet.count():
            self.currentElementIndex = 0

        self.currentElement = self.teachingSet.listOfElements[self.currentElementIndex]
        self.currentNormalizedInputs = list(self.currentElement.inputs)
        normalize(self.currentNormalizedInputs)

        self.examinedNeuron.learn(
            self.currentNormalizedInputs,
            self.currentElement.expectedOutput,
            teachingRatio,
            self.previousResponse,
            self.previousError
            )

        self.currentResponse = self.examinedNeuron.response(self.currentNormalizedInputs)
        self.currentError = self.currentElement.expectedOutput - self.currentResponse
        self.history.append(float(abs(self.currentError)))
        self.currentElementIndex += 1


print("1 neuron with 5 inputs and 1 output")
teachingSet = TeachingSet(5, 1)
teachingSet.listOfElements.append(Element([3, 4, 3, 4, 5], 1, "Typical object to accept"))
teachingSet.listOfElements.append(Element([1, -2, 1, -2, 4], -1, "Typical object to reject"))
teachingSet.listOfElements.append(Element([4, 2, 5, 3, 2], 0.8, "Untypical object to accept"))
teachingSet.listOfElements.append(Element([0, -1, 0, -3, -3], -0.8, "Untypical object to reject"))

neuron = Neuron()
neuron.randomize(-0.1, 0.1, 5)
pl = ProgramLogic(teachingSet, neuron)

print("How many times do you want to train the network?")
amount = int(input())
for i in range(amount):
    pl.performTeaching(0.5)

print("\n", pl.history)

