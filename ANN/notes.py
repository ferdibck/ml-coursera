import numpy as np

activation_functions = {
    "linear": {"function": lambda z: z, "derivative": lambda z: 1},
    "relu": {
        "function": lambda z: np.max(0, z),
        "derivative": lambda z: np.where(z > 0, 1, 0),
    },
    "leakyrelu": {
        "function": lambda z, a: z if z > 0 else a * z,
        "derivative": lambda z, a: 1 if z > 0 else a,
    },
    "binarystep": {
        "function": lambda z: 1 if z > 0 else 0,
        "derivative": lambda z: 1 if z == 0 else 0,
    },
    "sigmoid": {
        "function": lambda z: 1 / (1 + np.exp(-z)),
        "derivative": lambda z: 1 / (1 + np.exp(-z)) * (1 - 1 / (1 + np.exp(-z))),
    },
    "tanh": {
        "function": lambda z: np.tanh(z),
        "derivative": lambda z: 1 - (np.tanh(z)) ** 2,
    },
}
