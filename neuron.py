import numpy as np


class Neuron():

    def __init__(self, wValues: np.array, bValue: float) -> None:
        self.w = wValues
        self.b = bValue
    

    def zValue(self, xValues: np.array):
        print(f"w --> {self.w} // b --> {self.b}")
        print(f"x --> {xValues}")
        print("-----------------------")
        return np.dot(xValues, self.w) + self.b