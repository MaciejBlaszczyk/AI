from math import sqrt
from random import random


def normalize(signals):
    """Normalize list of signals (Euclidean norm)."""
    strength = 0
    for signal in signals:
        strength += signal * signal
    strength = sqrt(strength)
    for i in range(len(signals)):
        signals[i] /= strength


class NeuralNetwork:
    """Artificial neural network, consist of neurons."""

    def __init__(self, numberOfNeurons):
        self.numberOfNeurons = numberOfNeurons
        self.neurons = []
        for i in range(numberOfNeurons):
            self.neurons.append(Neuron())

    def randomize(self, min, max, inputCount):
        for neuron in self.neurons:
            neuron.randomize(min, max, inputCount)

    def learn(self, teachingElement, ratio, previousResponse, previousError):
        for i in range(self.numberOfNeurons):
            self.neurons[i].learn(
                teachingElement.inputs,
                teachingElement.expectedOutputs[i],
                ratio,
                previousResponse[i],
                previousError[i]
                )

    def response(self, signals):
        result = [float] * self.numberOfNeurons
        for i in range(self.numberOfNeurons):
            result[i] = self.neurons[i].response(signals)
        return result


class Neuron:
    """Artificial neuron - every example has its own weights which change in every learning step."""

    def __init__(self):
        self.weights = []

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


class Element:
    """Single element of the teaching set, consists of: list of inputs, expected output, comment."""

    def __init__(self, inputs, expectedOutputs, comment):
        self.inputs = inputs
        self.expectedOutputs = expectedOutputs
        self.comment = comment

    def show(self):
        print(self.comment)
        print("Inputs: ", self.inputs)
        print("Output: ", self.expectedOutputs)


class TeachingSet:
    """Teaching set, consist of elements list, number of inputs and outputs in every element."""

    def __init__(self, inputCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.listOfElements = []

    def normalize(self):
        for item in self.listOfElements:
            normalize(item.inputs)

    def show(self):
        for element in self.listOfElements:
            element.show()

    def count(self):
        return len(self.listOfElements)


class ProgramLogic:
    """Main logic of the network."""

    def __init__(self, teachingSet, network):
        self.normalizedTeachingSet = teachingSet
        teachingSet.normalize()
        self.examinedNetwork = network
        self.currentElementIndex = 0
        self.currentNormalizedElement = Element
        self.currentError = [float] * 3
        self.currentResponse = [float] * 3
        self.previousError = [[float] for _ in range(network.numberOfNeurons)]
        self.previousResponse = [[float] for _ in range(network.numberOfNeurons)]
        self.history = []

    def performTeaching(self, teachingRatio):
        """Conduct teaching network.
        Go element by element in the teaching set and teach network how to behave.
        By the way create history of errors during learning.
        """

        if self.currentElementIndex >= self.normalizedTeachingSet.count():
            self.currentElementIndex = 0

        self.currentNormalizedElement = self.normalizedTeachingSet.listOfElements[self.currentElementIndex]

        self.examinedNetwork.learn(
            self.currentNormalizedElement,
            teachingRatio,
            self.previousResponse,
            self.previousError
            )

        self.currentResponse = self.examinedNetwork.response(self.currentNormalizedElement.inputs)

        for i in range(len(self.currentResponse)):
            self.currentError[i] = self.currentNormalizedElement.expectedOutputs[i] - self.currentResponse[i]

        average = 0.0
        for elem in self.normalizedTeachingSet.listOfElements:
            resp = self.examinedNetwork.response(elem.inputs)
            for i in range(len(resp)):
                average += abs(elem.expectedOutputs[i] - resp[i])

        average /= len(self.examinedNetwork.neurons) * self.normalizedTeachingSet.count()

        self.history.append(float(abs(average)))
        self.currentElementIndex += 1


print("Linear network which contains 3 neurons with 5 inputs and 3 outputs.\n")

teachingSet = TeachingSet(5, 3)
teachingSet.listOfElements.append(Element([3, 4, 3, 4, 5], [1, -1, -1], "Typical object to accept by first neuron"))
teachingSet.listOfElements.append(Element([1, -2, 1, -2, 4], [-1, 1, -1], "Typical object to accept by second neuron"))
teachingSet.listOfElements.append(Element([-3, 2, -5, 3, 1], [-1, -1, 1], "Typical object to accept by third neuron"))
teachingSet.listOfElements.append(Element([4, 2, 5, 3, 2], [0.8, -1, -1], "Untypical object to accept by first neuron"))
teachingSet.listOfElements.append(Element([0, -1, 0, -3, -3], [-1, 0.8, -1], "Untypical object to accept by second neuron"))
teachingSet.listOfElements.append(Element([-5, 1, -1, 4, 2], [-1, -1, 0.8], "Untypical object to accept by third neuron"))
teachingSet.listOfElements.append(Element([-1, -1, -1, -1, -1], [-1, -1, -1], "Untypical object to reject by all neurons"))
print("Teaching set:\n")
teachingSet.show()

network = NeuralNetwork(3)
network.randomize(-0.1, 0.1, 5)

pl = ProgramLogic(teachingSet, network)

print("\nHow many times do you want to train the network?")
amount = int(input())
for i in range(amount):
    pl.performTeaching(0.1)

print("History of errors during training:")
for i in range(amount):
    print(pl.history[i])
