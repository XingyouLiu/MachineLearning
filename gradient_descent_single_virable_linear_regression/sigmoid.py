import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate z values
z = np.linspace(-10, 10, 100)

# Compute sigmoid values
sigma_z = sigmoid(z)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(z, sigma_z, '-r', label='sigmoid(z)')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.axvline(0, color='k', linestyle='--')
plt.axhline(y=0.5, color='k', linestyle='--')
plt.legend()
plt.show()
