# USO
# python test_network.py --model meme_not_meme.model --image /examples/xxx.xx

# importar los paquetes de keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# parseamos los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# cargamos la imagen
image = cv2.imread(args["image"])
orig = image.copy()

# pre-procesamos la imagen
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# cargamos el modelo de cnn
print("[INFO] loading network...")
model = load_model(args["model"])

# clasificamos el archivo cargado
(notSanta, santa) = model.predict(image)[0]

# construimos el label para mmstrar en el plot
label = "Meme" if santa > notSanta else "Not Meme"
proba = santa if santa > notSanta else notSanta
label = "{}: {:.2f}%".format(label, proba * 100)

# cargamos el label en la imagen
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# mostramos la imagen
cv2.imshow("Output", output)
cv2.waitKey(0)