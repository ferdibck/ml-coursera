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

    X = vectors[0]
    Y = vectors[1]

    return X, Y


def plot_data(X, Y):
    plt.plot(X, Y, marker="x", linestyle="")
    plt.show()


class simple_binary_classification:
    def __init__(self, X, Y, treshold) -> None:
        self.w = 0
        self.b = 0

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.m = len(X)
        self.treshold = treshold

    def predict_Y(self, X):
        z = self.w * X + self.b

        Y = 1 / (1 + np.exp(-z))

        return Y

    def calculate_cost(self):
        Y_pred = self.predict_Y(self.X_train)

        J = (
            -1
            / self.m
            * np.sum(
                self.Y_train * np.log(Y_pred) + (1 - self.Y_train) * np.log(1 - Y_pred)
            )
        )

        return J

    def gradient_descent(self, alpha, no_iterations):
        for _ in range(no_iterations):
            dJ_dw, dJ_db = self.compute_gradients()

            w_temp = self.w - alpha * dJ_dw
            b_temp = self.b - alpha * dJ_db

            self.w = w_temp
            self.b = b_temp

            J_curr = self.calculate_cost()
            print(J_curr)

    # Gradients aren't computed correctly
    def compute_gradients(self):
        Y_pred = self.predict_Y(self.X_train)

        dJ_dw = 1 / self.m * np.sum((Y_pred - self.Y_train) * self.X_train)
        dJ_db = 1 / self.m * np.sum((Y_pred - self.Y_train) ** 2)

        return dJ_dw, dJ_db

    def plot_model(self):
        max = np.max(self.X_train)
        min = np.min(self.X_train)

        x_values = np.linspace(min, max, 100)
        z = self.w * x_values + self.b
        y_values = 1 / (1 + np.exp(-z))

        data_to_plot = [
            (self.X_train, self.Y_train, "red", "x", "training", ""),
            (x_values, y_values, "blue", "", "model", "-"),
            (self.X_val, self.Y_val, "green", "o", "validation", ""),
        ]

        for x, y, color, marker, label, linestyle in data_to_plot:
            plt.plot(x, y, marker=marker, color=color, label=label, linestyle=linestyle)

        plt.ylabel("Category")
        plt.xlabel("x")
        plt.legend()
        plt.show()


data_path = "gpt_dataset.csv"
data = pd.read_csv(data_path)

X, Y = vectorization(data)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

model = simple_binary_classification(X_train, Y_train, 0.5)
model.gradient_descent(0.01, 100)
model.plot_model()
