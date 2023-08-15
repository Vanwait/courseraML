import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def descenso(xValues: np.array, yValue: np.array, w: np.array, b: float):

    m, n = xValues.shape
    desW = np.zeros(n)
    desB = 0.0

    for i in range(m):
        funcion = sigmoid(np.dot(w, xValues[i]) + b)
        for j in range(n):
            desW[j] += (funcion - yValue[i]) * xValues[i][j]
        desB += funcion - yValue[i]

    return desW / m, desB / m


def sigmoid(z: float):
    return 1 / (1 + np.exp(-z))


def descensoGradiente(xValues: np.array, yValues: np.array, w: np.array, b: np.array, alpha: float, numIt: int):
    cost = np.zeros(numIt)
    w = copy.deepcopy(w)
    for k in range(numIt):
        desW, desB = descenso(xValues, yValues, w, b)
        w = w - alpha * desW
        b = b - alpha * desB
        cost[k] = coste(xValues, yValues, w, b)
    plt.plot(cost)
    plt.show()
    return w, b


def coste(xValues: np.array, yValues: np.array, w: np.array, b: float):
    m, n = xValues.shape
    cost = 0
    for i in range(m):
        funcion = sigmoid(np.dot(xValues[i], w) + b)
        cost += -yValues[i] * np.log(funcion) - (1 - yValues[i]) * np.log(1 - funcion)
    return cost / m


def normalizacionZscore(xValues: np.array):
    return (xValues - np.mean(xValues, axis=0)) / np.std(xValues, axis=0)


def regresionLogica():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w_tmp  = np.zeros(len(X_train[0]))
    b_tmp  = 0.
    alph = 0.1
    iters = 10000

    w, b = descensoGradiente(X_train, y_train, w_tmp, b_tmp, alph, iters)
    print(f"w -> {w} // b -> {b}")


def regresionLogicaConEscalado():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    X_train = normalizacionZscore(X_train)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w_tmp  = np.zeros(len(X_train[0]))
    b_tmp  = 0.
    alph = 0.1
    iters = 10000

    w, b = descensoGradiente(X_train, y_train, w_tmp, b_tmp, alph, iters)
    print(f"w -> {w} // b -> {b}")


def regresioLogicaScikit():
    X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    print("Prediction on training set:", y_pred)
    print("Accuracy on training set:", lr_model.score(X, y))

    
if __name__ == '__main__':
    regresionLogica()
    print("---------------------------")
    regresionLogicaConEscalado()
    print("---------------------------")
    regresioLogicaScikit()



