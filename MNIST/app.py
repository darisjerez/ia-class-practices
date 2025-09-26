import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Precisión en test: {acc:.4f}")

indices = np.random.choice(len(x_test), 5)
images = x_test[indices]
labels = y_test[indices]

predictions = model.predict(images)

for i in range(5):
    plt.imshow(images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Etiqueta real: {labels[i]} | Predicho: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()