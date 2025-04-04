import numpy as np

# Define the number of samples for training and testing
num_train = 1000000
num_test = 20000  # 2% of training data

# Define Gaussian priors
sigma = [0.25, 0.5, 0.5, 0.5]  # Standard deviations

# Generate random x values
x_train = np.random.normal(0, sigma, (num_train, 4))
x_test = np.random.normal(0, sigma, (num_test, 4))

# Define arm lengths
l1, l2, l3 = 0.5, 0.5, 1.0

def forward_process(x):
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    y1 = x1 + l1 * np.sin(x2) + l2 * np.sin(x3 - x2) + l3 * np.sin(x4 - x2 - x3)
    y2 = l1 * np.cos(x2) + l2 * np.cos(x3 - x2) + l3 * np.cos(x4 - x2 - x3)
    return np.column_stack((y1, y2))

# Compute y values
y_train = forward_process(x_train)
y_test = forward_process(x_test)

# padd x_train and x_test to make them 10 dimension
x_train = np.pad(x_train, ((0, 0), (0, 6)), mode='constant', constant_values=0)
x_test = np.pad(x_test, ((0, 0), (0, 6)), mode='constant', constant_values=0)

# padd y_train and y_test to make them 8 dimension
y_train = np.pad(y_train, ((0, 0), (0, 6)), mode='constant', constant_values=0)
y_test = np.pad(y_test, ((0, 0), (0, 6)), mode='constant', constant_values=0)

# Save the generated data
np.save('data/x_train.npy', x_train)
np.save('data/y_train.npy', y_train)
np.save('data/x_test.npy', x_test)
np.save('data/y_test.npy', y_test)

print("Data generation complete. Files saved: x_train.npy, y_train.npy, x_test.npy, y_test.npy")
#print sample data
print("Sample x_train:", x_train[:5])
print("Sample y_train:", y_train[:5])