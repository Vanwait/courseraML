import numpy as np
import copy
import matplotlib.pyplot as plt

class RL():

    def __init__(self, initial_w, initial_b, iterations, X_train, y_train, alpha, normalizacion) -> None:
        if normalizacion:
            self.x = self.normalizacionZscore(X_train)
        else:
            self.x = X_train

        self.w = initial_w
        self.b = initial_b
        self.iter = iterations
        
        self.y = y_train
        self.alpha = alpha
    

    def descenso(self, w, b):
        m, n = self.x.shape

        sum_w = np.zeros(n, )
        sum_b = 0
        for i in range(m):
            funcion = np.dot(self.x[i], w) + b
            for j in range(n):
                sum_w[j] += (funcion - self.y[i]) * self.x[i][j]
            sum_b += funcion - self.y[i]
        return sum_w / m, sum_b / m
    

    def descensoGradiente(self):
        w = copy.deepcopy(self.w)
        b = copy.deepcopy(self.b)
        cost = np.zeros(self.iter)
        for k in range(self.iter):
            desW, desB = self.descenso(w, b)
            w = w - self.alpha * desW
            b = b - self.alpha * desB
            cost[k] = self.coste(w, b)
        plt.plot(cost)
        plt.show()
        self.w = w
        self.b = b
        return w, b
    

    def coste(self, w, b):
        cost = 0.0
        for i in range(len(self.x)):
            result = np.dot(self.x[i], w) + b
            cost += (result - self.y[i])**2

        return cost
    

    def normalizacionZscore(xValues: np.array):
        return (xValues - np.mean(xValues, axis=0)) / np.std(xValues, axis=0)