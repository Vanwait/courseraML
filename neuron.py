import numpy as np


class Neuron():

    def __init__(self, wValues: np.array, bValue: float) -> None:
        self.w = wValues
        self.b = bValue
    

    def zValue(self, xValues: np.array):
        return np.dot(xValues, self.w) + self.b