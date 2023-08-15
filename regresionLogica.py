import numpy as np
import copy
import matplotlib.pyplot as plt

class RLG():

    def __init__(self, initial_w, initial_b, iterations, X_train, y_train, alpha, normalizacion = False, lambd = 1) -> None:
        if normalizacion:
            self.x = self.normalizacionZscore(X_train)
        else:
            self.x = X_train

        self.w = initial_w
        self.b = initial_b
        self.iter = iterations
        
        self.y = y_train
        self.alpha = alpha
        self.lambd = lambd
    

    def descenso(self, w, b):
        m, n = self.x.shape
        desW = np.zeros(n)
        desB = 0.0

        for i in range(m):
            funcion = self.sigmoid(np.dot(w, self.x[i]) + b)
            for j in range(n):
                desW[j] += (funcion - self.y[i]) * self.x[i][j]
            desB += funcion - self.y[i]
        
        desW /= m
        desB /= m

        for j in range(n):
            desW[j] = desW[j] + (self.lambd / m) * w[j]

        return desW, desB
    

    def descensoGradiente(self):
        w = copy.deepcopy(self.w)
        b = copy.deepcopy(self.b)
        cost = np.zeros(self.iter)
        m, n = self.x.shape
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
        m, n = self.x.shape
        cost = 0
        for i in range(m):
            funcion = self.sigmoid(np.dot(self.x[i], w) + b)
            cost += -self.y[i] * np.log(funcion) - (1 - self.y[i]) * np.log(1 - funcion)
        
        cost /= m
        regCost = 0
        for j in range(n):
            regCost += (w[j] ** 2)

        regCost = (self.lambd / (2*m)) * regCost
        return cost + regCost
            

    def normalizacionZscore(self, xValues: np.array):
        return (xValues - np.mean(xValues, axis=0)) / np.std(xValues, axis=0)
    

    def sigmoid(self, z: float):
        return 1 / (1 + np.exp(-z))