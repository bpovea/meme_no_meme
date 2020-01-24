# configuramos matplotlib para que los plots puedan ser guardados en background
import matplotlib
matplotlib.use("Agg")
# habilitamos que se puedan mostrar los graficos en el notebook
import matplotlib.pyplot as plt

# Importamos los paquetes necesarios
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os

# Inicializamos algunas variables que usaremos para cargar/guardar
# los archivos
dataset = 'images'
model_file = 'meme_not_meme.model'
plot_file = 'plot.png'

# Inicializamos el número de epocas para entrenar, la tasa de entrenamiento
# y el batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# Inicializamos los datos y labels
print("[INFO] loading images...")
data = []
labels = []

# Guardamos el path de las imagenes y lo aleatorizamos
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# Recorremos el path de imágenes
for imagePath in imagePaths:
	# cargamos las imágenes, preprocesamos y las almacenamos
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	# Obtenemos el label de la clase segun el nombre del path
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "meme" else 0
	labels.append(label)

# Estandarizamos los pixeles para que esten en el rango [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# dividimos la data en training y testing. 75% para entrenar
# y el otro 25% para pruebas
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convertimos los labels de enteros a vectores
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Construimos el generador de imagenes para aumentar los datos
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# Inicializamos el modelo
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Entrenamiento de la red
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# Se guarda el modelo en memoria
print("[INFO] serializing network...")
model.save(model_file)

# plot de loss training y accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Meme/Not Meme")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_file)
plt.show()

# importamos py de matriz de confusion
from matriz_confusion import graficar_matriz_de_confusion

# graficamos en base a los resultados obtenidos vs los esperados
y_pred = model.predict_classes(testX)
y_ref = np.argmax(testY,axis=1)
etiquetas = ['0','1']
graficar_matriz_de_confusion(y_ref, y_pred, etiquetas)