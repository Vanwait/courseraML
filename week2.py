import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


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
    cost = np.zeros(numIt)
    for k in range(numIt):
        desW, desB = descenso(xValues, yValues, w, b)
        w = w - alpha * desW
        b = b - alpha * desB
        cost[k] = coste(xValues, yValues, w, b)
    plt.plot(cost)
    plt.show()
    return w, b


def normalizacionZscore(xValues: np.array):
    return (xValues - np.mean(xValues, axis=0)) / np.std(xValues, axis=0)


def coste(xValues: np.array, yValues: np.array, w: np.array, b: float):
    cost = 0.0
    for i in range(len(xValues)):
        result = np.dot(xValues[i], w) + b
        cost += (result - yValues[i])**2

    return cost


def regresiónLinealMain():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    initial_w = np.zeros(len(X_train[0]))
    initial_b = 0.
    iterations = 1000
    alpha = 5.0e-7
    w, b = descensoGradiente(initial_w, initial_b, iterations, X_train, y_train, alpha, descenso)
    print(f"El valor de w es {w}, el valor de b es {b}")
    for i in range(len(X_train)):
        print(f"Predicción -> {np.dot(w, X_train[i]) + b}")
        print(f"Valor real -> {y_train[i]}")
    print("--------------------------------------")

    alpha = 5.0e-2
    X_trainN = normalizacionZscore(X_train)
    w, b = descensoGradiente(initial_w, initial_b, iterations, X_trainN, y_train, alpha, descenso)
    print(f"El valor de w es {w}, el valor de b es {b}")
    for i in range(len(X_train)):
        print(f"Predicción -> {np.dot(w, X_trainN[i]) + b}")
        print(f"Valor real -> {y_train[i]}")


def mostrarValues(x, X, y, w, b, feature):
    if feature:
        plt.title("feature engineering")
    else:
        plt.title("no feature engineering")
    plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
    plt.plot(x,X@w + b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()


def ingenieriaCaracteristicas():
    x = np.arange(0, 20, 1)
    y = x**2
    #X = normalizacionZscore(x.reshape(-1, 1))
    #X = x**2
    #X = normalizacionZscore(X.reshape(-1, 1))
    #X = X.reshape(-1, 1)
    X = np.c_[x, x**2, x**3]
    X = normalizacionZscore(X)

    w, b = descensoGradiente(np.zeros(X.shape[1]), 0, 10000, X, y, 1e-1, descenso) 
    print(f"{w} // {b}") ##Viendo los valores de w, al tener más valor para x^2, esta variable tiene más relevancia en este caso, lo cual es normal dado que la función es y = x^2
    mostrarValues(x, X, y, w, b, True)


def scikitRegresion():
    x = np.arange(0, 20, 1)
    y = 2 * x + 13 
    y = y / 43
    y = y * x
    X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7]
    #X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    sgdr = SGDRegressor(max_iter=10000)
    sgdr.fit(X_norm, y)

    b = sgdr.intercept_
    w = sgdr.coef_
    print(f"{w} // {b}")
    y_pred_sgd = sgdr.predict(X_norm)
    mostrarValues(x, X, y, w, b, True)


    xnor = normalizacionZscore(X)
    w, b = descensoGradiente(np.zeros(X.shape[1]), 0, 10000, xnor, y, 1e-1, descenso) 
    mostrarValues(x, X, y, w, b, True)

    print(f"Prediction on training set:\n{y_pred_sgd[:4]}" )
    print(f"Target values \n{y[:4]}")
    print(f"{w} // {b}")
    

if __name__ == "__main__":
    #regresiónLinealMain()
    #ingenieriaCaracteristicas()
    scikitRegresion()
    
