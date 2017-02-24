from random import random
from math import exp, sqrt
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Artificial neural network, consist of layers which contain neurons."""

    def __init__(self, inputCount, numberOfLayers, numberOfNeuronsInLayer):
        self.inputCount = inputCount
        self.numberOfLayers = numberOfLayers
        self.numberOfNeuronsInLayer = numberOfNeuronsInLayer
        self.layers = [[Neuron(self.inputCount) for _ in range(numberOfNeuronsInLayer)] for _ in range(numberOfLayers)]
        self.bias = -1


class Neuron:
    """Artificial neuron - every example has its own weights which change in every learning step."""

    def __init__(self, neuronInputs):
        self.prevWeights = [0. for _ in range(neuronInputs)]
        self.weights = []
        self.randomize(-1, 1, neuronInputs)
        self.temp = []

    def randomize(self, min, max, inputCount):
        """Set neuron weights to random numbers between min and max."""
        length = max - min
        for i in range(inputCount):
            self.weights.append(min + length * random())

    def learn(self, signals, error, ratio, momentum):
        """Set neuron weights depending on ratio, momentum, input signals, previous errors and weights."""
        self.temp = list(self.weights)
        for i in range(len(self.weights)):
            self.weights[i] += ratio * error * signals[i] - momentum * (self.weights[i] - self.prevWeights[i])
        self.prevWeights = self.temp

    def response(self, signals):
        """Return neuron's response (sum of multiplied weights and input signals)."""
        result = 0.
        for signal, weight in zip(signals, self.weights):
            result += signal * weight
        return result

    def activationFunction(self, arg):
        """Return sigmoid function value."""
        beta = 1.0
        return 1.0 / (1.0 + exp(-beta * arg))


class ProgramLogic:
    """Main logic of the network."""

    def __init__(self, network):
        self.examinedNetwork = network
        self.inputs = [float] * (network.inputCount - 1)
        self.expectedOutputs = [float] * network.numberOfNeuronsInLayer
        self.ratio = 0.7
        self.momentum = 0.3
        self.responses = [[network.bias for _ in range(network.numberOfNeuronsInLayer + 1)] for _ in range(network.numberOfLayers)]
        self.errors = [[0. for _ in range(network.numberOfNeuronsInLayer)] for _ in range(network.numberOfLayers)]
        self.temp = [float] * network.numberOfNeuronsInLayer
        self.history = []

    def performTeaching(self):
        """Conduct teaching network."""

        self.inputs = [5 * random() - 2.5 for _ in range(network.inputCount - 1)]  # -1 because of bias presence in higher layers
        self.inputs.append(network.bias)
        if self.inputs[0] > 0:
            self.expectedOutputs[0] = 1.0
            self.expectedOutputs[1] = 0.0
        else:
            self.expectedOutputs[0] = 0.0
            self.expectedOutputs[1] = 1.0

        # Neurons outputs:
        for i in range(self.examinedNetwork.numberOfNeuronsInLayer):
            tmp = self.examinedNetwork.layers[0][i].response(self.inputs)
            self.responses[0][i] = self.examinedNetwork.layers[0][i].activationFunction(tmp)

        for i in range(1, self.examinedNetwork.numberOfLayers):
            for j in range(len(self.examinedNetwork.layers[i])):
                tmp = self.examinedNetwork.layers[i][j].response(self.responses[i - 1])
                self.responses[i][j] = self.examinedNetwork.layers[i][j].activationFunction(tmp)

        # Errors(backpropagation algorithm) and mean-squared error:
        for i in range(self.examinedNetwork.numberOfNeuronsInLayer):
            resp = self.responses[self.examinedNetwork.numberOfLayers - 1][i]
            self.temp[i] = self.expectedOutputs[i] - resp
            self.errors[self.examinedNetwork.numberOfLayers - 1][i] = self.temp[i] * resp * (1 - resp)

        mse = 0.
        for i in range(self.examinedNetwork.numberOfNeuronsInLayer):
            mse += self.temp[i] * self.temp[i]
        mse = sqrt(mse / 2.)
        self.history.append(mse)
        if self.temp[0] < 0.5 and self.temp[1] < 0.5:
            print("GOOD")
            pass
        else:
            print("BAD")
            pass

        for i in range(self.examinedNetwork.numberOfLayers - 2, -1, -1):
            for j in range(len(self.examinedNetwork.layers[i])):
                resp = self.responses[i][j]
                self.temp[j] = 0.
                for k in range(self.examinedNetwork.numberOfNeuronsInLayer):
                    self.temp[j] += self.errors[i + 1][k] * self.examinedNetwork.layers[i + 1][k].weights[j]
                self.errors[i][j] = self.temp[j] * resp * (1 - resp)

        # Learning:
        for i in range(self.examinedNetwork.numberOfNeuronsInLayer):
            self.examinedNetwork.layers[0][i].learn(
                self.inputs,
                self.errors[0][i],
                self.ratio,
                self.momentum
                )

        for i in range(1, self.examinedNetwork.numberOfLayers):
            for j in range(len(self.examinedNetwork.layers[i])):
                self.examinedNetwork.layers[i][j].learn(
                    self.responses[i - 1],
                    self.errors[i][j],
                    self.ratio,
                    self.momentum
                    )


network = NeuralNetwork(3, 2, 2)

pl = ProgramLogic(network)

counter = input("How much times do you want to train the network?(more than 300 is recommended to show precisly error decreasing)")
print("Answers of the network after every learning iteration:")
for i in range(int(counter)):
    pl.performTeaching()
print("History of mean-squared errors:")
plt.plot(pl.history, 'bo')
plt.show()
