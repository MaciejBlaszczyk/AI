from math import sqrt


class Neuron:
    weights = []
    prevWeights = []

    def __init__(self):
        pass

    def normalize(self, signals):
        strength = 0
        for signal in signals:
            strength += signal * signal
        strength = sqrt(strength)

        for signal in signals:
            signal /= strength


class Element:
    inputs = []
    expectedOutputs = []
    comment = ""

    def __init__(self, inputs, expectedOutputs, comment):
        self.inputs = inputs
        self. expectedOutputs = expectedOutputs
        self.comment = comment

    def clone(self):
        elem = Element()
        elem.inputs = self.inputs
        elem.expectedOutputs = self.expectedOutputs
        return elem


class TeachingSet:
    listOfElements = []
    inputCount = 0.
    outputCount = 0.

    def __init__(self, inputCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount

    def normalize(self):
        for item in self.listOfElements:
            Neuron.normalize(Neuron(), item.inputs)

    def readFromConsole(self):
        print("Enter number of inputs:")
        self.inputCount = int(input())
        print("Enter number of outputs:")
        self.outputCount = int(input())
        pass


print("1 neuron with 5 inputs and 1 output")
teachingSet = TeachingSet(5, 1)
teachingSet.listOfElements.append(Element([3, 4, 3, 4, 5], [1], "Typical object to accept"))
teachingSet.listOfElements.append(Element([1, -2, 1, -2, 4], [-1], "Typical object to reject"))
teachingSet.listOfElements.append(Element([4, 2, 5, 3, 2], [0.8], "Untypical object to accept"))
teachingSet.listOfElements.append(Element([3, 4, 3, 4, 5], [-0.8], "Untypical object to reject"))







