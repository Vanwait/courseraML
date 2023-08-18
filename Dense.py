import numpy as np

class Dense2():


    def __init__(self, W, b) -> None:
        self.w = W
        self.b = b

    
    def zValue(self, x):
        try:
            units = self.w.shape[1]
            a = np.zeros(units)
            for j in range(units):
                a[j] = self.sigmoid(np.dot(x, self.w[:, j]) + self.b[j])
            return a
        except Exception as e:
            print(x)
            print(self.w[:, j])
            print(self.b)
            print(e)

    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
