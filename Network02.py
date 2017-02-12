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
featureNames = ["number of legs ", "lives in water", "can fly       ", "has feathers  ", "egg-laying    "]

weights = [
    [ 4.00,  0.01, 0.01, -1.00, -1.50],
    [ 2.00, -1.00, 2.00,  2.50,  2.00],
    [-1.00,  3.50, 0.01, -2.00,  1.50]
]

network = LinearNetwork(weights)

print("                     " + "       ".join(classNames))
for i in range(5):
    print()
    print(featureNames[i], end = "")
    for j in range(3):
        print( "        " + str(weights[j][i]), end = "")

features = []
questions = ["How many legs does it have?", "Does it live in water?", "Can it fly?",
             "Does it have feathers?", "Does it lay eggs?"]
print("\nEnter features of your animal:")
print("If you want to answet the question \"yes\", write 1, if \"no\", write -1, "
      "if you are not sure, write a number between -1 and 1 ")
for i in range(5):
    print(questions[i])
    features.append(float(input()))

mammal = 0.
bird = 0.
fish = 0.
for i in range(5):
    mammal += weights[0][i] * features[i]
    bird += weights[1][i] * features[i]
    fish += weights[2][i] * features[i]

if mammal > bird and mammal > fish and mammal > 2:
    print("It is a mammal")
if bird > mammal and bird > fish and bird > 2:
    print("It is a bird")
if fish > mammal and fish > bird and fish > 2:
    print("It is a fish")

