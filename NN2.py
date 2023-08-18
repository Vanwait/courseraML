import numpy as np
from Dense import Dense2

class NN2():


    def __init__(self, wValues: list, bValues: list) -> None:
        self.layers = []
        for layer in range(len(wValues)):
            self.layers.append(Dense2(wValues[layer], bValues[layer]))
    

    def forwardProp(self, x):
        a = x
        for layer in self.layers:
            a = layer.zValue(a)
        return a

    