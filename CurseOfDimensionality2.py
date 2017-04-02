#The aim of the assignment is to investigate how the following quantities change depending on the number of dimensions. For the purposes of this task we use the Euclidean distance.

# We have a hypercube with edges length equal to 1.0. We randomly fill it with evenly distributed points.
# What is the ratio between the standard deviation of distance between those points and the average distance between them?

import numpy as np
import math
import matplotlib.pyplot as plt

ratio = []
numberOfPoints = 100
numberOfDimensions = 101

for dimensions in range(2, numberOfDimensions):
    points = np.array([np.random.uniform(0, 1, size = dimensions) for _ in range(numberOfPoints)])
    distances = np.empty(int((numberOfPoints * (numberOfPoints - 1)) / 2))

    temp = 1
    position = 0
    for i in range(numberOfPoints):
        for j in range(temp, numberOfPoints):
            distances[position] = (math.sqrt(math.fsum((points[i] - points[j]) ** 2)))
            position += 1
        temp += 1

    mean = math.fsum(distances)/len(distances)
    stddev = math.sqrt((math.fsum((distances - mean) ** 2))/len(distances))
    ratio.append(stddev/mean)

plt.plot([_ for _ in range(2, numberOfDimensions)], ratio, 'bo')
plt.axis([0, 101, 0, 0.5])
plt.ylabel('Ratio')
plt.xlabel('Number Of Dimensions')
plt.title('Ratio (Mean / Standard Deviation) in dependance from amount of dimensions')
plt.show()
