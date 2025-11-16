import numpy as np

np.random.seed(0)
w_true = np.random.uniform(2, 10, size=3)
b_true = np.random.uniform(2, 10)
N = 1000

X = np.random.rand(N, 3)
y = X @ w_true + b_true
X_ext = np.hstack((X, np.ones((N, 1))))

w_est = np.zeros(4)
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    indices = np.random.permutation(N)
    X_shuffled, y_shuffled = X_ext[indices], y[indices]

    for i in range(N):
        x_i = X_shuffled[i].reshape(1, -1)
        y_i = y_shuffled[i]
        grad_i = 2 * x_i.T @ (x_i @ w_est - y_i)
        w_est -= learning_rate * grad_i.flatten()

    mse = np.mean((X_ext @ w_est - y) ** 2)
    print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.6f}")

w_hat, b_hat = w_est[:-1], w_est[-1]
print("\n=== Stochastic Gradient Descent Results ===")
print(f"True weights:      {np.round(w_true, 4)}, bias = {b_true:.4f}")
print(f"Estimated weights: {np.round(w_hat, 4)}, bias = {b_hat:.4f}")
