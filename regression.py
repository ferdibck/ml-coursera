import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_data(n, m, t):
    x_values = np.random.uniform(0, 1000, n)
    y_values = []

    for xi in x_values:
        y = m * xi + t + np.random.normal(0, 3, n)

        y_values.append(y)

    data = [x_values, y_values]

    return data


class model:
    def __init__(self):
        self.w = 0
        self.b = 0

    def predict_X(self, X):
        Y = []

        for xi in X:
            y = self.w * xi + self.b

        Y.append(y)

        return Y

    def cost_function(self, data):
        X_train = data[0]
        Y_train = data[1]

        Y_pred = self.predict_X(self, X_train)

        sum = 0

        for i in range(len(Y_train)):
            sum += (Y_pred[i] - Y_train[i]) ** 2

        cost = 1 / (2 * len(Y_train)) * sum
