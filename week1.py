
import numpy as np

def descenso(xValues: list, yValues: list, w: int, b: int):
    sum_w = 0
    sum_b = 0
    m = len(xValues)
    for pos in range(m):
        funcion = w * xValues[pos] + b
        sum_w += (funcion - yValues[pos]) * xValues[pos]
        sum_b += (funcion - yValues[pos])
        
    return sum_w / m, sum_b / m

def descensoGradiente(w_ini: int, b_ini: int, descenso, numIt: int, alpha: float, xValues: list, yValues: list):
    w = w_ini
    b = b_ini
    for _ in range(numIt):
        desW, desB = descenso(xValues, yValues, w, b)
        w = w - alpha * desW
        b = b - alpha * desB
    return w, b

if __name__ == "__main__":
    w_init = 0
    b_init = 0
    iterations = 10000
    tmp_alpha = 1.0e-2
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    w, b = descensoGradiente(w_init, b_init, descenso, iterations, tmp_alpha, x_train ,y_train)
    print(f"Valor de la función de predicción f(x) = {w} * x + {b}")