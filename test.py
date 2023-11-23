import numpy as np
import matplotlib.pyplot as plt


# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Generate X values (input)
X = np.linspace(-10, 10, 100)  # Example range of X values

# Set values for w and b
w = 1  # Example weight
b = 0  # Example bias

# Calculate Z
Z = w * X + b

# Calculate Y using the sigmoid function
Y = sigmoid(Z)

# Plot the sigmoid curve
plt.figure(figsize=(8, 6))
plt.plot(X, Y)
plt.title("Sigmoid Function")
plt.xlabel("Input (X)")
plt.ylabel("Output (Y)")
plt.grid(True)
plt.axhline(y=0.5, color="r", linestyle="--", label="Threshold")
plt.legend()
plt.show()
