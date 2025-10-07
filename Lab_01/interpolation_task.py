import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def f(x):
    return x**2 + x - 4

step = 0.25
x = np.arange(-5, 5 + 1e-12, step)
y = f(x)

lin = interp1d(x, y, kind="linear")
cub = interp1d(x, y, kind="cubic")

x_s = np.linspace(x.min(), x.max(), 400)
y_lin = lin(x_s)
y_cub = cub(x_s)

plt.figure(figsize=(8,5))
plt.plot(x, y, "o", label="initial points")        
plt.plot(x_s, y_lin, "-", label="linear interp.") 
plt.plot(x_s, y_cub, "--", label="cubic interp.")
plt.title("f(x) = x^2 + x - 4  (linear vs cubic interp.)")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.legend(); plt.grid(True)
plt.show()
