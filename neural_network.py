import numpy as np
from abc import ABC, abstractmethod


class neural_network:
    def __init__(self, input_size):
        self.first_layer = output_layer()
        self.input_size = input_size

    def add_layer(self, no_units):
        self.first_layer = self.first_layer.add_layer(no_units)

    def init_model(self):
        self.first_layer.init_model(self.input_size)

    def inference(self, X):
        return self.first_layer.inference(X)


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
    def init_unit(self):
        pass

    @abstractmethod
    def inference(self):
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

        self.init_unit()

        self.next_layer.init_model(self.size)

    def init_unit(self):
        for i in range(self.size):
            self.units[i] = unit(self.input_size)

    def inference(self, X):
        activation_vector = np.empty(self.size)
        for i in range(self.size):
            activation_i = self.units[i].compute_activation(X)
            activation_vector[i] = activation_i

        output = self.next_layer.inference(activation_vector)

        return output


class output_layer(layer_element):
    def __init__(self):
        self.units = np.empty([1], dtype=object)

    def add_layer(self, no_units):
        return layer(no_units, self)

    def init_model(self, input_size):
        self.input_size = input_size

        self.init_unit()

    def init_unit(self):
        self.units[0] = unit(self.input_size)

    def inference(self, input):
        output = self.units[0].compute_activation(input)
        return output


class unit:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def compute_activation(self, X):
        Y = 1 / (1 + np.exp(-(np.dot(self.weights, X) + self.bias)))
        return Y


model = neural_network(2)
model.add_layer(2)
model.init_model()
print(model.inference([0, 0]))
