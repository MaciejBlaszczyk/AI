#The aim of the assignment is to investigate how the following quantities change depending on the number of dimensions. For the purposes of this task we use the Euclidean distance.

# We have a hyperball with a radius equal to 1.0 inscribed inside a hypercube with edges length equal to 2.0.
# Hyperball in multidimensional space is defined as a set of points whose distance from its centre is no greater than its radius.
# We randomly fill the hypercube with evenly distributed points. What % of those points would be inside the hyperball, and what % outside – in the corners?


import numpy as np
import math
import matplotlib.pyplot as plt

numberOfPoints = 1000
numberOfDimensions = 21
percentResults = []

for dimensions in range(2, numberOfDimensions):
    points = np.array([np.random.uniform(-1, 1, size = dimensions) for _ in range(numberOfPoints)])

    squaredPoints = points ** 2

    distances = np.array([math.sqrt(math.fsum(squaredPoints[i])) for i in range(numberOfPoints)])

    results = np.array([1 if distances[i] < 1 else 0 for i in range(numberOfPoints)])

    percent = (math.fsum(results)/numberOfPoints)*100
    print("Dimensions:", dimensions, " ", percent, "% points in hyperball")
    percentResults.append(percent)


plt.plot([_ for _ in range(2, numberOfDimensions)], percentResults)
plt.axis([0, 21, -10, 110])
plt.xlabel('Dimensions')
plt.ylabel('% of points in hypersphere')
plt.title('% of points in hypersphere in dependance from amount of dimensions')
plt.show()
