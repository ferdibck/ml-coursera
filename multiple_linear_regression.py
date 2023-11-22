import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def vectorization(df):
    col_vectors = []

    for col in df.columns:
        col_data = df[col].values
        col_vectors.append(col_data)

    vectors = np.array(col_vectors)

    X = vectors[1]
    Y = vectors[2]

    return X, Y


def plot_data(X, Y):
    plt.plot(X, Y, marker="x", linestyle="")
    plt.show()


class multiple_linear_regression:
    def __init__(self, X, Y):
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.m = len(X_train)

        self.w = np.zeros(self.m)
        self.b = 0
