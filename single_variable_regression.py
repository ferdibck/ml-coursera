import numpy as np
import matplotlib.pyplot as plt


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

    def predict_Y(self, X):
        Y_pred = self.w * X + self.b

        return Y_pred

    def calculate_cost(self, X, Y):
        Y_pred = self.predict_Y(X)

        squared_error = (Y_pred - Y) ** 2
        error_sum = np.sum(squared_error)

        cost = 1 / (2 * len(X)) * error_sum

        return cost

    def gradient_descent(self, alpha, epsilon):
        change = epsilon + 1
        while change > epsilon:
            pass

    def scale_feature(X):
        mean = np.mean(X)
        max = np.max(X)
        min = np.min(X)

        X = (X - mean) / (max - min)
