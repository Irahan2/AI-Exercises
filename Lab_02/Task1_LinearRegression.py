import numpy as np

np.random.seed(0)
w_true = np.random.uniform(2, 10, size=3)
b_true = np.random.uniform(2, 10)

N = 1000
X = np.random.rand(N, 3)
y = X @ w_true + b_true

X_ext = np.hstack((X, np.ones((N, 1))))

w_est = np.linalg.pinv(X_ext.T @ X_ext) @ X_ext.T @ y


w_hat = w_est[:-1]
b_hat = w_est[-1]

y_pred = X_ext @ w_est
mse = np.mean((y - y_pred) ** 2)

print("=== Linear Regression (Closed-Form) ===")
print(f"True weights:      {np.round(w_true, 4)},  bias = {b_true:.4f}")
print(f"Estimated weights: {np.round(w_hat, 4)},  bias = {b_hat:.4f}")
print(f"Mean Squared Error: {mse:.8f}")
