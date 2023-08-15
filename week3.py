import numpy as np
import copy
import matplotlib.pyplot as plt

def descenso(xValues: np.array, yValue: np.array, w: np.array, b: float):

    m, n = xValues.shape
    desW = np.zeros(n)
    desB = 0

    for i in range(m):
        funcion = sigmoid(np.dot(w[i], xValues[i]) + b)
        for j in range(n):
            desW[j] += (funcion - yValue[i]) * xValues[i][j]
        desB += funcion - yValue[i]

    return desW / m, desB / m


def sigmoid(z: float):
    return 1 / 1 + np.exp(-z)


def descensoGradiente(xValues: np.array, yValues: np.array, w: np.array, b: np.array, alpha: float, numIt: int, descenso):
    cost = np.zeros(numIt)
    for k in range(numIt):
        desW, desB = descenso(xValues, yValues, w, b)
        w = w - alpha * desW
        b = b - alpha * desB
        cost[k] = coste(xValues, yValues, w, b)
    plt.plot(cost)
    plt.show()


def coste(xValues: np.array, yValues: np.array, w: np.array, b: np.array):
    m, n = xValues.shape
    cost = 0
    for i in range(m):
        funcion = sigmoid(np.dot(xValues, w) + b)
        cost += -yValues[i] * np.log(funcion) - (1 - yValues[i]) * np.log(1 - funcion)
    return cost / m


def normalizacionZscore(xValues: np.array):
    return (xValues - np.mean(xValues, axis=0)) / np.std(xValues, axis=0)


if __name__ == '__main__':
    pass


