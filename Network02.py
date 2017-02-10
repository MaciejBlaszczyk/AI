
class Neuron:
    weights = []
    prevWeights = []


class LinearNetwork:
    neurons = []

    def __init__(self, initialWeights):
        numberOfNeurons = len(initialWeights)
        for i in range(numberOfNeurons - 1):
            self.neurons.append(Neuron())
            self.neurons[i].weights = initialWeights[i]


classNames = ["mammal", "bird", "fish"]
featureNames = ["number of legs", "lives in water", "can fly", "has feathers", "egg-laying"]

weights = [
    [ 4, 0.01, 0.01,  -1, -1.5],
    [ 2,   -1,    2, 2.5,    2],
    [-1,  3.5, 0.01,  -2,  1.5]
]

network = LinearNetwork(weights)

print(weights)