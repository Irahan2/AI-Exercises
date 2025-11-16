import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)
y_test_cat = keras.utils.to_categorical(y_test, 10)

def add_salt_pepper_noise(images, amount=0.15):
    noisy = images.copy()
    num_pixels = int(amount * images.shape[1] * images.shape[2])
    for i in range(images.shape[0]):
        for _ in range(num_pixels):
            x, y = np.random.randint(0, 28, 2)
            noisy[i, x, y, 0] = 1 if np.random.rand() < 0.5 else 0
    return noisy

x_test_noisy = add_salt_pepper_noise(x_test, amount=0.15)

# Visualize noisy images
plt.figure(figsize=(9, 3))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

model = keras.models.load_model('cnn_mnist_model.h5') if tf.io.gfile.exists('cnn_mnist_model.h5') else None

if model is None:
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    model.fit(x_train, y_train_cat, epochs=3, batch_size=128)
    model.save('cnn_mnist_model.h5')

# Evaluate on noisy data
loss, acc = model.evaluate(x_test_noisy, y_test_cat)
print(f"Accuracy on noisy data: {acc:.4f}")