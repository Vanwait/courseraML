import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Input
from keras import Sequential
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.activations import sigmoid


if __name__ == '__main__':
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
