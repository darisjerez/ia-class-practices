import os
import tensorflow as tf
from keras import layers, models
import numpy as np
import cv2
from keras.models import load_model
'''
# Tamaño de imagen y ruta
width = 300 
height = 300
ruta_train = 'train/'

train_x = []
train_y = []

extensiones_validas = ('.jpg', '.jpeg', '.png')

# Carga de imagenes 
for i in os.listdir(ruta_train):
    subfolder_path = os.path.join(ruta_train, i)
    if not os.path.isdir(subfolder_path):
        continue

    for j in os.listdir(subfolder_path):
        if not j.lower().endswith(extensiones_validas):
            continue

        img_path = os.path.join(subfolder_path, j)
        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            continue

        try:
            resized_image = cv2.resize(img, (width, height))
            train_x.append(resized_image)

            if i == 'cats':
                train_y.append([0, 1])
            else:
                train_y.append([1, 0])
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")

x_data = np.array(train_x)
y_data = np.array(train_y)

# Normalizar imágenes
x_data = x_data / 255.0

# Definir modelo
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), input_shape=(width, height, 3)), 
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(32, (3,3)), 
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(64, (3,3)), 
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),
    layers.Dense(64),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(2),
    layers.Activation('sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 100
model.fit(x_data, y_data, epochs=epochs)

# Guardar modelo
models.save_model(model, 'MiModelo.keras')
'''

width = 300
height = 300

# Cargar el modelo .keras
modelo = load_model('MiModelo.keras')

ruta_imagen = 'train\dogs\dog_7.jpg'

# Verificar existencia y extensión válida
extensiones_validas = ('.jpg', '.jpeg', '.png')
if not os.path.exists(ruta_imagen):
    print(f"La imagen no existe: {ruta_imagen}")
elif not ruta_imagen.lower().endswith(extensiones_validas):
    print("Formato de imagen no válido. Usa .jpg, .jpeg o .png.")
else:
    # Leer y procesar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("No se pudo leer la imagen. Asegúrate de que el archivo es una imagen válida.")
    else:
        img = cv2.resize(img, (width, height))
        img = img / 255.0
        img = np.expand_dims(img, axis=0) 

        prediccion = modelo.predict(img)[0]

        # Interpretar Resultado
        clases = ['Perro🐶', 'Gato🐈']
        indice_clase = np.argmax(prediccion)
        porcentaje = prediccion[indice_clase] * 100

        print(f"Imagen: {ruta_imagen}")
        print(f"Predicción: {clases[indice_clase].capitalize()} ({porcentaje:.2f}%)")
