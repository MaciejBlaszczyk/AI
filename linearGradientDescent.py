import pandas
import numpy as np
import matplotlib.pyplot as plt


def cost(theta, x, y):
    cost = 0.
    a = theta[1]
    b = theta[0]
    N = len(x)

    for i in range(N):
        cost += (a * x[i] + b - y[i]) ** 2

    return cost / N


def gradientStep(theta, x, y, alpha):
    a = theta[1]
    b = theta[0]
    N = len(x)
    aDerivative = 0.
    bDerivative = 0.

    for i in range(N):
        aDerivative += (a * x[i] + b - y[i]) * x[i] * (2. / N)
        bDerivative += (a * x[i] + b - y[i]) * (2. / N)

    a = a - alpha * aDerivative
    b = b - alpha * bDerivative
    newTheta = [b, a]
    return newTheta


def gradientDescent(theta, x, y, alpha, steps):
    for i in range(steps):
        if i % 100 == 0:
            print("Current cost: ", cost(theta, x, y))
        theta = gradientStep(theta, x, y, alpha)

    return theta


data = pandas.read_csv("ex1.csv", header=None)
X = data[0]
Y = data[1]
N = len(X)

initialTheta = np.zeros(2)

shuffledX = []
shuffledY = []
shuffleIndex = [i for i in range(len(X))]
np.random.shuffle(shuffleIndex)
for i in shuffleIndex:
    shuffledX.append(X[i])
    shuffledY.append(Y[i])

xTrain = np.array(shuffledX[:int(N*3/4)])
yTrain = np.array(shuffledY[:int(N*3/4)])
xTest = np.array(shuffledX[int(N*3/4):])
yTest = np.array(shuffledY[int(N*3/4):])

theta = gradientDescent(initialTheta, xTrain, yTrain, 0.00001, 1000)

print("Training cost: ", cost(theta, xTrain, yTrain))
print("Test cost: ", cost(theta, xTest, yTest))


plt.plot(X, Y, 'b.')

estX = np.linspace(min(X), max(X))
estY = [theta[1]*x + theta[0] for x in estX]
plt.plot(estX, estY, 'r')
plt.show()