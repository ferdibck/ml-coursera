import numpy as np
import matplotlib.pyplot as plt


def generate_data(n, m, t):
    x_values = np.random.uniform(0, 100, n)
    y_values = m * x_values + t + np.random.normal(0, t * 7.5, n)

    return x_values, y_values


def plot_data(x, y):
    plt.scatter(x, y)
    plt.show()


class model:
    def __init__(self):
        self.w = np.array([0])
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
            cost_curr = self.calculate_cost()
            cost_prev = cost_curr
            w_temp = self.w - alpha * dJ_dw
            b_temp = self.b - alpha * dJ_db

            self.w = w_temp
            self.b = b_temp

    def scale_feature(X):
        mean = np.mean(X)
        max = np.max(X)
        min = np.min(X)

        X = (X - mean) / (max - min)
