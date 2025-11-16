import matplotlib.pyplot as plt
from tensorflow import keras

(x_train, y_train), _ = keras.datasets.mnist.load_data()

plt.figure(figsize=(9, 9))
for i in range(81):
    plt.subplot(9, 9, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
    plt.title(str(y_train[i]), fontsize=8)

plt.suptitle("MNIST 9x9 Grid", fontsize=16)
plt.tight_layout()
plt.show()
