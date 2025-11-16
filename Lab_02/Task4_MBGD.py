import numpy as np

np.random.seed(0)
w_true = np.random.uniform(2, 10, size=3)
b_true = np.random.uniform(2, 10)
N = 1000

X = np.random.rand(N, 3)
y = X @ w_true + b_true
X_ext = np.hstack((X, np.ones((N, 1))))   # (1000,4)

w_est = np.zeros(4)
learning_rate = 0.05
epochs = 10
batch_size = 25
y_orig = y.copy()

for epoch in range(epochs):
    data = np.hstack((X_ext, y.reshape(-1, 1)))
    np.random.shuffle(data)
    x_shuffled, y_shuffled = data[:, :4], data[:, 4]

    for j in range(N // batch_size):
        start, end = j * batch_size, (j + 1) * batch_size
        X_batch = x_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        grad_batch = (2 / batch_size) * X_batch.T @ (X_batch @ w_est - y_batch)
        w_est -= learning_rate * grad_batch

    mse = np.mean((X_ext @ w_est - y_orig) ** 2)
    print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.6f}")

w_hat, b_hat = w_est[:-1], w_est[-1]
print("\n=== Mini-Batch Gradient Descent Results ===")
print(f"True weights:      {np.round(w_true, 4)}, bias = {b_true:.4f}")
print(f"Estimated weights: {np.round(w_hat, 4)}, bias = {b_hat:.4f}")
