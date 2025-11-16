import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
w_true = np.random.uniform(2, 10, size=3)
b_true = np.random.uniform(2, 10)
N = 1000

X = np.random.rand(N, 3)
y = X @ w_true + b_true
X_ext = np.hstack((X, np.ones((N, 1))))

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

w_closed = np.linalg.pinv(X_ext.T @ X_ext) @ X_ext.T @ y

def batch_gd(X, y, lr=0.05, epochs=10):
    N, d = X.shape
    w = np.zeros(d)
    hist = []
    for _ in range(epochs):
        grad = (2 / N) * X.T @ (X @ w - y)
        w -= lr * grad
        hist.append(mse(y, X @ w))  
    return w, hist

def sgd(X, y, lr=0.01, epochs=10):
    N, d = X.shape
    w = np.zeros(d)
    hist = []
    for _ in range(epochs):
        idx = np.random.permutation(N)
        Xs, ys = X[idx], y[idx]
        for i in range(N):
            xi = Xs[i:i+1]
            yi = ys[i]
            grad = 2 * xi.T @ (xi @ w - yi)
            w -= lr * grad.flatten()
        hist.append(mse(y, X @ w))
    return w, hist

def mbgd(X, y, lr=0.05, epochs=10, batch_size=25):
    N, d = X.shape
    w = np.zeros(d)
    hist = []
    for _ in range(epochs):
        data = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data)
        Xs, ys = data[:, :d], data[:, d]
        for j in range(N // batch_size):
            s, e = j * batch_size, (j + 1) * batch_size
            Xb, yb = Xs[s:e], ys[s:e]
            grad = (2 / batch_size) * Xb.T @ (Xb @ w - yb)
            w -= lr * grad
        hist.append(mse(y, X @ w))
    return w, hist

w_bgd, hist_bgd = batch_gd(X_ext, y)
w_sgd, hist_sgd = sgd(X_ext, y)
w_mbgd, hist_mbgd = mbgd(X_ext, y)


plt.figure(figsize=(8, 5))
plt.plot(hist_bgd, label="Batch GD", linewidth=2)
plt.plot(hist_sgd, label="SGD", linewidth=2)
plt.plot(hist_mbgd, label="Mini-Batch GD", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Gradient Descent Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\n=== Weight Vector Comparison (w_true, w_R, w_BGD, w_SGD, w_MBGD) ===")
print("w_true     :", np.round(np.hstack((w_true, b_true)), 4))
print("w_R (closed):", np.round(w_closed, 4))
print("w_BGD      :", np.round(w_bgd, 4))
print("w_SGD      :", np.round(w_sgd, 4))
print("w_MBGD     :", np.round(w_mbgd, 4))


print("\n=== Summary ===")
print(f"True weights: {np.round(w_true,4)}, bias = {b_true:.4f}")
print(f"Final MSE (BGD):  {hist_bgd[-1]:.6f}")
print(f"Final MSE (SGD):  {hist_sgd[-1]:.6f}")
print(f"Final MSE (MBGD): {hist_mbgd[-1]:.6f}")
