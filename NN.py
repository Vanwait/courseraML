import numpy as np
from neuron import Neuron
import copy



class NN():

    def __init__(self, numLayers: int, numUnits: np.array, wValues: np.array, bValues: np.array) -> None:
        if numLayers != len(numUnits):
            raise Exception("Se tiene que especificar el número de neuronas por capa")
        self.layers = []
        for layer in range(numLayers):
            ly = []
            for unit in range(numUnits[layer]):
                ly.append(Neuron(wValues[layer][unit], bValues[layer][unit]))
            self.layers.append(ly)
        #self.mostrar()


    def forwardProp(self, xValues: np.array):
        a = copy.deepcopy(xValues)
        for layer in self.layers:
            aAux = []
            for unit in layer:
                aAux.append(self.sigmoid(unit.zValue(a)))
            a = np.array(aAux)
        return a

    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

    def mostrar(self):
        for layer in range(len(self.layers)):
            print(f"Estoy en la Capa {layer}")

            for unit in range(len(self.layers[layer])):
                print(f"En la Neurona {unit} -->")
                print(f"wValues --> {self.layers[layer][unit].w} // bValue --> {self.layers[layer][unit].b}")



