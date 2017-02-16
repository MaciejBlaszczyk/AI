class Neuron:

    def __init__(self, initialWeights):
        self.weights = initialWeights


class LinearNetwork:
    """network which contains neurons"""
    def __init__(self, initialWeights):
        self.numberOfNeurons = len(initialWeights)
        self.neurons = []
        for i in range(self.numberOfNeurons):
            self.neurons.append(Neuron(initialWeights[i]))


classNames = ["mammal", "bird", "fish"]
featureNames = ["number of legs ", "lives in water", "can fly       ", "has feathers  ", "egg-laying    "]
weights = [
    [ 4.00,  0.01, 0.01, -1.00, -1.50],
    [ 2.00, -1.00, 2.00,  2.50,  2.00],
    [-1.00,  3.50, 0.01, -2.00,  1.50]]
features = []
questions = ["How many legs does it have?", "Does it live in water?", "Can it fly?",
             "Does it have feathers?", "Does it lay eggs?"]


print("                     " + "       ".join(classNames))
for i in range(5):
    print()
    print(featureNames[i], end = "")
    for j in range(3):
        print( "        " + str(weights[j][i]), end = "")
print("\nEnter features of your animal:")
print("If you want to answer the question \"yes\", write 1, if \"no\", write -1, "
      "if you are not sure, write a number between -1 and 1 ")


network = LinearNetwork(weights)

for i in range(5):
    print(questions[i])
    features.append(float(input()))

mammal = 0.
bird = 0.
fish = 0.

for i in range(5):
    mammal += network.neurons[0].weights[i] * features[i]
    bird += network.neurons[1].weights[i] * features[i]
    fish += network.neurons[2].weights[i] * features[i]

if mammal > bird and mammal > fish and mammal > 2:
    print("It is a mammal")
if bird > mammal and bird > fish and bird > 2:
    print("It is a bird")
if fish > mammal and fish > bird and fish > 2:
    print("It is a fish")

