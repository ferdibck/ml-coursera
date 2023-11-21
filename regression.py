import numpy as np
import matplotlib.pyplot as plt


def generate_data(n, m, t):
    x_values = np.random.uniform(0, 100, n)
    y_values = m * x_values + t + np.random.normal(0, 5, n)

    X = x_values.reshape(-1, 1)  # Reshaping to create a column vector
    Y = y_values.reshape(-1, 1)  # Reshaping to create a column vector

    return X, Y


class model:
    def __init__(self, X_train, Y_train):
        # Training data
        self.X_train = X_train
        self.Y_train = Y_train

        self.m = len(X_train)

        # Initial parameters
        self.w = 0
        self.b = 0

    def scale_feature(self, X):
        mean = np.mean(X)
        max = np.max(X)
        min = np.min(X)

        X = (X - mean) / (max - min)

        return X

    def predict_Y(self, X):
        Y_pred = self.w * X + self.b

        return Y_pred

    def gradient_descent(self, n_iterations):
        alpha = 0.0003

        cost_values = []
        i_values = []

        for i in range(n_iterations):
            cost = self.calculate_cost()
            cost_values.append(cost)
            i_values.append(i)

            dJ_dw, dJ_db = self.compute_gradient()

            new_w = self.w - alpha * dJ_dw
            new_b = self.b - alpha * dJ_db

            self.w = new_w
            self.b = new_b

        plt.plot(i_values, cost_values)
        plt.show()

        print(f"w: {self.w}, b: {self.b}")

    def compute_gradient(self):
        Y_pred = self.predict_Y(self.X_train)

        dJ_dw = 0
        dJ_db = 0

        dJ_dw = 1 / (self.m) * np.sum((Y_pred - self.Y_train) * self.X_train)
        dJ_db = 1 / (self.m) * np.sum(Y_pred - self.Y_train)

        return dJ_dw, dJ_db

    def calculate_cost(self):
        Y_pred = self.predict_Y(self.X_train)

        squared_error = (Y_pred - self.Y_train) ** 2
        error_sum = np.sum(squared_error)

        cost = 1 / (2 * len(self.X_train)) * error_sum

        print(cost)

        return cost

    def plot_model_and_data(self):
        plt.scatter(self.X_train, self.Y_train, label="Data Points")
        plt.xlabel("X values")
        plt.ylabel("Y values")

        # Plotting the linear function
        x_range = np.linspace(np.min(self.X_train), np.max(self.X_train), 100)
        y_range = self.w * x_range + self.b
        plt.plot(x_range, y_range, color="red", label="Linear Function")

        plt.title("Linear Regression")
        plt.legend()
        plt.show()

    def plot_contour(self):
        # Generating values for w and b
        w_values = np.linspace(self.w - 100, self.w + 100, 100)
        b_values = np.linspace(self.b - 100, self.b + 100, 100)

        # Creating a meshgrid for w and b
        W, B = np.meshgrid(w_values, b_values)

        # Calculating the cost for each combination of w and b
        cost_values = np.zeros((len(w_values), len(b_values)))

        for i in range(len(w_values)):
            for j in range(len(b_values)):
                Y_pred = W[i, j] * self.X_train + B[i, j]
                squared_error = (Y_pred - self.Y_train) ** 2
                cost_values[i, j] = 1 / (2 * len(self.X_train)) * np.sum(squared_error)

        # Plotting the contour plot
        plt.figure(figsize=(8, 6))
        contours = plt.contour(W, B, cost_values, levels=50)
        plt.colorbar(contours)
        plt.scatter(self.w, self.b, color="red", marker="*", label="Optimal w, b")
        plt.xlabel("w")
        plt.ylabel("b")
        plt.title("Contour Plot of Cost Function")
        plt.legend()
        plt.show()


X, Y = generate_data(300, 2.5, 10)

model = model(X, Y)

model.gradient_descent(50)

model.plot_model_and_data()
model.plot_contour()
