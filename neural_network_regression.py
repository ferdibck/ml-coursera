import numpy as np
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd


class neural_network:
    def __init__(self, input_size, output_size):
        self.first_layer = output_layer(output_size)
        self.input_size = input_size

    def add_layer(self, no_units):
        self.first_layer = self.first_layer.add_layer(no_units)

    def init_model(self):
        self.first_layer.init_model(self.input_size)

    def inference(self, X):
        return self.first_layer.inference(X)

    def train(self, X_train, Y_train, learning_rate, num_epochs):
        batch_size = len(X_train) / num_epochs

        for epoch in range(num_epochs):
            total_loss = 0
            y_preds = []
            for i in range(len(X_train)):
                y_pred = self.inference(X[i])
                total_loss += (y_pred - Y[i]) ** 2
                y_preds.append(y_pred)

            total_loss *= 1 / (2 * len(Y_train))

            print(f"Loss {epoch}: ", total_loss)

            self.backpropagation2(X_train, Y_train, y_preds, learning_rate)

    def backpropagation(self, y_pred, y, learning_rate):
        loss = 1 / 2 * (y_pred - y) ** 2
        dl_da = y_pred - y

        self.first_layer.backpropagation(dl_da, learning_rate)


class layer_element(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add_layer(self, no_units):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_units(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def backpropagation(self):
        pass


class layer(layer_element):
    def __init__(self, no_units, next_layer):
        self.size = no_units
        self.units = np.empty(no_units, dtype=object)

        self.next_layer = next_layer

    def add_layer(self, no_units):
        self.next_layer = self.next_layer.add_layer(no_units)

        return self

    def init_model(self, input_size):
        self.input_size = input_size

        self.init_units()

        self.next_layer.init_model(self.size)

    def init_units(self):
        for i in range(self.size):
            self.units[i] = unit(self.input_size)

    def inference(self, X):
        activation_vector = np.empty(self.size)
        for i in range(self.size):
            activation_i = self.units[i].compute_activation(X)
            activation_vector[i] = activation_i

        self.last_a = activation_vector

        output = self.next_layer.inference(activation_vector)

        return output

    def backpropagation(self, dl_dprev, learning_rate):
        da_dz = np.where(self.last_a > 0, 1, 0)
        dl_dprev = np.full(len(da_dz), dl_dprev)
        dl_dprev *= da_dz

        for i in range(self.size):
            self.units[i].backpropagation(dl_dprev[i], learning_rate)


class output_layer(layer_element):
    def __init__(self, no_units):
        self.size = no_units
        self.units = np.empty(no_units, dtype=object)

    def add_layer(self, no_units):
        return layer(no_units, self)

    def init_model(self, input_size):
        self.input_size = input_size

        self.init_units()

    def init_units(self):
        for i in range(self.size):
            self.units[i] = unit(self.input_size)

    def inference(self, X):
        activation_vector = np.empty(self.size)
        for i in range(self.size):
            activation_i = self.units[i].compute_activation(X)
            activation_vector[i] = activation_i

        self.last_a = activation_vector

        return activation_vector

    def backpropagation(self, dl_da, learning_rate):
        da_dz = np.where(self.last_a > 0, 1, 0)
        dl_prev = dl_da * da_dz

        for i in range(self.size):
            self.units[i].backpropagation(dl_prev[i], learning_rate)

        return dl_prev


class unit:
    def __init__(self, input_size):
        # self.weights = np.zeros(input_size)
        self.input_size = input_size
        self.weights = np.random.rand(input_size)
        self.bias = 0

    def compute_activation(self, X):
        self.last_X = X
        Z = np.dot(self.weights, X.reshape(-1, 1)) + self.bias
        Y = self.ReLU(Z)
        self.last_a = Y
        return Y

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def backpropagation(self, dl_prev, learning_rate):
        dz_dw = self.last_X
        dz_db = 1

        dl_dw = dl_prev * dz_dw
        dl_db = dl_prev * dz_db

        self.gradient_descent(dl_dw, dl_db, learning_rate)

    def gradient_descent(self, dl_dw, dl_db, learning_rate):
        self.weights -= learning_rate * dl_dw
        self.bias -= learning_rate * dl_db

        # print("New bias", self.bias)
        # print("New weights", self.weights)


def vectorization(df):
    X = df.drop(["Performance Index", "Extracurricular Activities"], axis=1)
    Y = df["Performance Index"]

    X = X.to_numpy()
    Y = Y.to_numpy()

    return X, Y


def plot_data(X, Y, y_pred):
    plt.plot(X, Y, marker="x", linestyle="")
    plt.plot(X, y_pred, marker="o", color="red", linestyle="-")
    plt.show()


data_path = "Student_performance.csv"
data = pd.read_csv(data_path)

X, Y = vectorization(data)

model = neural_network(4, 1)
model.add_layer(16)
model.add_layer(8)
model.add_layer(10)
model.init_model()

for i in range(10):
    print(i)
    loss = 0
    for i in range(len(X)):
        y_pred = model.inference(X[i, :])
        model.backpropagation(y_pred, Y[i], 1 * 10 ** (-9))

        loss += abs(y_pred - Y[i])
        loss /= len(Y)

    print(loss)

for i in range(len(X)):
    y_pred = model.inference(X[i])

    print(X[i], Y[i], y_pred)
