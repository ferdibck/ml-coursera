import numpy as np
from abc import ABC, abstractmethod

class neural_network:
  def __init__(self):
    self.first_layer = output_layer()
  
  def add_layer(self, no_units):
      self.first_layer = self.first_layer.add_layers(no_units)

class layer_element(ABC):
  @abstractmethod
  def __init__(self):
    pass

  @abstractmethod
  def add_layer(self, no_units):
    pass

class layer(layer_element):
  def __init__(self, no_units, next_layer):
    self.units = np.array(no_units, dtype=object)

    for i in range(no_units):
      self.units[i] = unit()

    self.next_layer = next_layer

    return self
  
  def add_layer(self, no_units):
    self.next_layer = self.next_layer.add_layer(no_units)

class output_layer(layer_element):
  def __init__(self):
    self.units = np.array(1)
    self.units[0] = unit()

  def add_layer(self, no_units):
    return layer(no_units, self)
  
class unit:
  def __init__(self):
    