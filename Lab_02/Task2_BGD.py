import numpy as np

np.random.seed(0)
w_true = np.random.uniform(2, 10, size=3)
b_true = np.random.uniform(2, 10)
N = 1000

X = np.random.rand(N, 3)
y = X @ w_true + b_true
X_ext = np.hstack((X, np.ones((N, 1))))

w_est = np.zeros(4)
learning_rate = 0.05   # biraz daha hızlı yakınsama
epochs = 10

for epoch in range(epochs):
    grad = (2 / N) * X_ext.T @ (X_ext @ w_est - y)
    w_est -= learning_rate * grad
    mse = np.mean((X_ext @ w_est - y) ** 2)
    print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.6f}")

w_hat, b_hat = w_est[:-1], w_est[-1]
print("\n=== Batch Gradient Descent Results ===")
print(f"True weights:      {np.round(w_true, 4)}, bias = {b_true:.4f}")
print(f"Estimated weights: {np.round(w_hat, 4)}, bias = {b_hat:.4f}")
