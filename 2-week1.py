import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Input
from keras import Sequential
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.activations import sigmoid
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from NN import NN
from NN2 import NN2

def parteUno():
    X_train = np.array([[1.0], [2.0]], dtype=np.float32)
    Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

    linearLayer = Dense(units=1, activation='linear')

    a1 = linearLayer(X_train[0].reshape(1, 1))
    w, b = linearLayer.get_weights()
    print(f"w -> {w} // b -> {b}")
    linearLayer.set_weights([np.array([[200]]), np.array([100])])
    w, b = linearLayer.get_weights()
    print(f"w -> {w} // b -> {b}")
    prediction_tf = linearLayer(X_train)
    prediction_np = np.dot(X_train, w) + b
    print(f"Tensorflow --> {prediction_tf}")
    print(f"Numpy --> {prediction_np}")

    print("-------------------------")

    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)
    Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)

    model = Sequential([Dense(1, input_dim=1, activation='sigmoid', name='L1')])

    LGLayer = model.get_layer('L1')
    LGLayer.set_weights([np.array([[2]]), np.array([-4.5])])

    fore = model.predict(X_train[0].reshape(1, 1))
    w, b = LGLayer.get_weights()
    fore2 = 1 / (1 + np.exp(np.dot(w, X_train[0].reshape(1, 1)) + b))
    print(fore)
    print(fore2)


def RoastCoffee1():
    x = np.array([[200.0, 17.0]])
    layer_1 = Dense(units=3, activation='sigmoid')
    layer_2 = Dense(units=1, activation='sigmoid')
    a1 = layer_1(x)
    print(a1)
    a2 = layer_2(a1)
    if a2 >= 0.5:
        print(str(1))
    else:
        print(str(0))


def RoastCoffee2():
    x = np.array([[200.0, 17.0], [120.0, 5.0], [425.0, 20.0], [121.0, 18.0]])
    y = np.array([1, 0, 0, 1])
    model = Sequential([
        Dense(units=3, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')])
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.01))
    model.fit(x, y, epochs=10)
    fore = model.predict(np.array([[120.0, 5.0]]))
    print(fore)
    if fore >= 0.5:
        print(str(1))
    else:
        print(str(0))


def RoastCoffee3():
    wValues = [
        [np.array([-0.03889755, 0.87865]), np.array([-0.05489755, 0.097678865]), np.array([0.0344755, -0.23637865])], 
        [np.array([-0.06811755, 0.55637865, 0.09873])]
    ]
    bValues = [
        [-1, 1, 2], 
        [3]
    ]
    model = NN(2, np.array([3, 1]), wValues, bValues)
    fore = model.forwardProp(np.array([120.0, 5.0]))
    print(f"El resultado es --> {fore}")


def RoastCoffee4():
    wVal = [np.array([
        [-0.03889755, -0.05489755, 0.0344755],
        [0.87865, 0.097678865, -0.23637865]
    ]), np.array([[-0.06811755], [0.55637865], [0.09873]])]
    bVal = [np.array([-1, 1, 2]), np.array([3])]
    model = NN2(wVal, bVal)
    fore = model.forwardProp(np.array([120.0, 5.0]))
    print(f"El resultado es --> {fore}")

if __name__ == '__main__':
    print("----------NN1----------")
    RoastCoffee3()
    print("----------NN2----------")
    RoastCoffee4()
    #prueba()
