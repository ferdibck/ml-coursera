import numpy as np
from abc import ABC, abstractmethod


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

    def compute_cost(self, Xs, Y):
        cost = 0

        for i, X in enumerate(Xs):
            Y_pred = self.inference(X)
            cost += Y[i] * np.log(Y_pred) + (1 - Y[i]) * np.log(1 - Y_pred)

        cost /= -(len(Y))

        return cost

    def train(self, Xs_train, Ys_train, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            pass

    def backpropagation(self, y_pred, y):
        loss = -y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)

        self.first_layer.backpropagation()


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

        output = self.next_layer.inference(activation_vector)

        return output

    def backpropagation(self):
        pass


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

        return activation_vector

    def backpropagation(self):
        pass


class unit:
    def __init__(self, input_size):
        # self.weights = np.zeros(input_size)
        self.weights = np.random.rand(input_size)
        self.bias = 0

    def compute_activation(self, X):
        Z = np.dot(self.weights, X) + self.bias
        Y = 1 / (1 + np.exp(-Z))
        self.last_a = Y
        return Y


model = neural_network(2, 3)
model.add_layer(2)
model.init_model()
print(model.inference([100, 100]))
