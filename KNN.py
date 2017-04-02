from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

iris = datasets.load_iris()
points = np.array(iris.data[ : , :2])
types = iris.target


EucK1noCNN = np.array(types)
EucK5noCNN = np.array(types)
ManK1noCNN = np.array(types)
ManK5noCNN = np.array(types)


plt.figure(1)
plt.scatter(points[:, 0], points[:, 1], c=types)
distancesEuc = np.array(distance.cdist(points, points, 'euclidean'))
distancesEuc[distancesEuc == 0] = np.nan
distancesMan = np.array(distance.cdist(points, points, 'euclidean'))
distancesMan[distancesMan == 0] = np.nan


for i in range(len(points)):
    minIndex = np.nanargmin(distancesEuc[i])
    EucK1noCNN[i] = types[minIndex]
    minIndex = np.nanargmin(distancesMan[i])
    ManK1noCNN[i] = types[minIndex]
plt.figure(2)
plt.scatter(points[:, 0], points[:, 1], c=EucK1noCNN)
plt.figure(3)
plt.scatter(points[:, 0], points[:, 1], c=ManK1noCNN)


for i in range(len(points)):
    tempResults = []
    for _ in range(5):
        minIndex = np.nanargmin(distancesEuc[i])
        distancesEuc[i][minIndex] = np.nan
        tempResults.append(types[minIndex])
    if tempResults.count(0) > 2:
        EucK5noCNN[i] = 0
    elif tempResults.count(1) > 2:
        EucK5noCNN[i] = 1
    elif tempResults.count(2) > 2:
        EucK5noCNN[i] = 2
    else:
        EucK5noCNN[i] = 3

    tempResults = []
    for _ in range(5):
        minIndex = np.nanargmin(distancesMan[i])
        distancesMan[i][minIndex] = np.nan
        tempResults.append(types[minIndex])
    if tempResults.count(0) > 2:
        ManK5noCNN[i] = 0
    elif tempResults.count(1) > 2:
        ManK5noCNN[i] = 1
    elif tempResults.count(2) > 2:
        ManK5noCNN[i] = 2
    else:
        ManK5noCNN[i] = 3
plt.figure(4)
plt.scatter(points[:, 0], points[:, 1], c=EucK5noCNN)
plt.figure(5)
plt.scatter(points[:, 0], points[:, 1], c=ManK5noCNN)

plt.show()


