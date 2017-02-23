from random import random
from math import exp


class NeuralNetwork:
    """Artificial neural network, consist of neurons."""

    def __init__(self, inputCount, numberOfLayers, numberOfNeuronsInLayer):
        self.inputCount = inputCount
        self.numberOfLayers = numberOfLayers
        self.numberOfNeuronsInLayer = numberOfNeuronsInLayer
        self.layers = [[Neuron(self.inputCount) for _ in range(numberOfNeuronsInLayer)] for _ in range(numberOfLayers)]
        self.bias = -1

    def learn(self, teachingElement, ratio, previousResponse, previousError):
        pass


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

    def activationFunction(self, arg):
        beta = 1.0
        return 1.0 / (1.0 + exp(arg * beta))


class ProgramLogic:
    """Main logic of the network."""

    def __init__(self, network):
        self.examinedNetwork = network
        self.inputs = [float] * (network.inputCount - 1)
        self.expectedOutputs = [float] * network.numberOfNeuronsInLayer
        self.inputs.append(network.bias)
        self.ratio = 0.1
        self.responses = [[network.bias for _ in range(network.numberOfNeuronsInLayer + 1)] for _ in range(network.numberOfLayers)]
        self.errors = [[0. for _ in range(network.numberOfNeuronsInLayer)] for _ in range(network.numberOfLayers)]

    def performTeaching(self):
        """Conduct teaching network."""
        self.inputs = [5 * random() - 2.5 for _ in range(network.inputCount - 1)] #-1 because of bias presence in higher layers
        if self.inputs[0] > 0:
            self.expectedOutputs[0] = 1.0
            self.expectedOutputs[1] = 0.0
        else:
            self.expectedOutputs[0] = 0.0
            self.expectedOutputs[1] = 1.0

        for i in range(self.examinedNetwork.numberOfNeuronsInLayer):
            temp = self.examinedNetwork.layers[0][i].response(self.inputs)
            self.responses[0][i] = self.examinedNetwork.layers[0][i].activationFunction(temp)

        for i in range(1, self.examinedNetwork.numberOfLayers):
            for j in range(len(self.examinedNetwork.layers[i])):
                temp = self.examinedNetwork.layers[i][j].response(self.responses[i-1])
                self.responses[i][j] = self.examinedNetwork.layers[i][j].activationFunction(temp)

        for i in range(self.examinedNetwork.numberOfNeuronsInLayer):
            resp = self.responses[self.examinedNetwork.numberOfLayers - 1][i]
            temp = self.expectedOutputs[i] - resp
            self.errors[self.examinedNetwork.numberOfLayers - 1][i] = temp * resp * (1 - resp)

        for i in range(self.examinedNetwork.numberOfLayers - 2, -1, -1):
            for j in range(len(self.examinedNetwork.layers[i])):
                resp = self.responses[i][j]
                temp = 0.
                for k in range(self.examinedNetwork.numberOfNeuronsInLayer):
                    temp += self.responses[i + 1][k] * self.examinedNetwork.layers[i + 1][k].weights[j]
                self.errors[i][j] = temp * resp * (1 - resp)

        for i in range(1, self.examinedNetwork.numberOfLayers):
            for j in range(len(self.examinedNetwork.layers[i])):
                self.examinedNetwork.layers[i][j].learn(

                )


network = NeuralNetwork(3, 2, 2)

pl = ProgramLogic(network)

counter = input("How much times do you want to train the network?")
for i in range(int(input)):
    pl.performTeaching()