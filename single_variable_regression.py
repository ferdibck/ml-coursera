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
    def __init__(self, X, Y):
        self.w = np.array([0])
        self.b = 0
        self.X_train = X
        self.Y_train = Y
        self.m = len(X)

    def predict_Y(self, X):
        Y_pred = self.w * X + self.b

        return Y_pred

    def calculate_cost(self, X, Y):
        Y_pred = self.predict_Y(X)

        squared_error = (Y_pred - Y) ** 2
        error_sum = np.sum(squared_error)

        cost = 1 / (2 * len(X)) * error_sum

        return cost

    def gradient_descent(self, alpha, iterations):
        cost_values = []
        i_values = []

        i = 0
        while i < iterations:
            cost = self.calculate_cost(self.X_train, self.Y_train)""
            cost_values.append(cost)
            i_values.append(i)

            dJ_dw, dJ_db = self.calculate_gradient()

            w_temp = self.w - alpha * dJ_dw
            b_temp = self.b - alpha * dJ_db

            self.w = w_temp
            self.b = b_temp

            i += 1

        plt.plot(i_values, cost_values)
        plt.show()

    def scale_feature(self, X):
        mean = np.mean(X)
        max = np.max(X)
        min = np.min(X)

        X = (X - mean) / (max - min)

        return X

    def calculate_gradient(self):
        Y_pred = self.predict_Y(self.X_train)

        dJ_dw = 1 / (self.m) * np.sum((Y_pred - self.Y_train) * self.X_train)
        dJ_db = 1 / (self.m) * np.sum(Y_pred - self.Y_train)

        return dJ_dw, dJ_db


X, Y = generate_data(100, 5, 10)

plot_data(X, Y)

model = model(X, Y)

model.gradient_descent(0.01, 100)
