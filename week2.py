import numpy as np
import copy

def makePrediction(xValues: np.array, w: np.array, b: float):
    return np.dot(xValues, w) + b


def descenso(xValues: np.array, yValues: np.array, w: np.array, b: float):

    m, n = xValues.shape

    sum_w = np.zeros(n, )
    sum_b = 0
    for i in range(m):
        funcion = np.dot(xValues[i], w) + b
        for j in range(n):
            sum_w[j] += (funcion - yValues[i]) * xValues[i][j]
        sum_b += funcion - yValues[i]
    return sum_w / m, sum_b / m



def descensoGradiente(w_init: np.array, b_init: np.array, numIt: int, xValues: np.array, yValues: np.array, alpha: float, descenso):
    w = copy.deepcopy(w_init)
    b = copy.deepcopy(b_init)

    for _ in range(numIt):
        desW, desB = descenso(xValues, yValues, w, b)
        w = w - alpha * desW
        b = b - alpha * desB
    return w, b


if __name__ == "__main__":
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    initial_w = np.zeros(len(X_train[0]))
    initial_b = 0.
    iterations = 1000
    alpha = 5.0e-7
    w, b = descensoGradiente(initial_w, initial_b, iterations, X_train, y_train, alpha, descenso)
    print(f"El valor de w es {w}, el valor de b es {b}")
    for i in range(len(X_train)):
        print(f"Predicción -> {np.dot(w, X_train[i]) + b}")
        print(f"Valor real -> {y_train[i]}")