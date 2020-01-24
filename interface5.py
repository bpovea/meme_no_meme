from tkinter import *

# importing os module
import os

# importing shutil module
import shutil

# importar los paquetes de keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

root=Tk()

miFrame=Frame(root,width=1200, height=5200)
miFrame.pack()

pathLabel=Label(miFrame,text="Path:",font=("Courier", 20))
pathLabel.grid(row=0,column=0,sticky="e",padx=10,pady=5)

#var para almacenar el path
tkpath=StringVar()

cuadroPath=Entry(miFrame, textvariable=tkpath,width=50,font=("Courier", 10))
cuadroPath.grid(row=0,column=1)

#var para resultado
result=StringVar()

resultLabel=Label(miFrame,textvariable=result,font=("Courier", 20))
resultLabel.grid(row=1,column=0,padx=10,pady=5,columnspan=2)

# Codigo para el boton
def codigoBoton():
	# /home/bpovea/workspace/IA/image-classification-keras/test
	path = tkpath.get()
	destination_path = path+'/memes'
	memes = 0
	not_memes = 0
	# cargamos el modelo de cnn
	model = load_model('meme_not_meme.model')
	try:
		# Create target Directory
		os.mkdir(destination_path)
		print("Directory " , destination_path ,  " Created ")
	except FileExistsError:
		print("Directory " , destination_path ,  " already exists")
	# List files and directories
	for image in os.listdir(path):
		image = '/'+image
		source = path+image
		if source != destination_path:
			# cargamos la imagen
			keras_image = cv2.imread(source)
			# pre-procesamos la imagen
			keras_image = cv2.resize(keras_image, (28, 28))
			keras_image = keras_image.astype("float") / 255.0
			keras_image = img_to_array(keras_image)
			keras_image = np.expand_dims(keras_image, axis=0)
			# clasificamos el archivo cargado
			(notSanta, santa) = model.predict(keras_image)[0]
			es_meme = True if santa > notSanta else False
			if es_meme:
				dest = shutil.move(source, destination_path+image)
				# print(dest)
				memes += 1
			else:
				# print('  ---> '+source)
				not_memes += 1
	result.set('Memes encontrados: '+str(memes)+'\nNo Memes: '+str(not_memes)+'\nImagenes procesadas: '+str(memes+not_memes))

# Crear botones
botonProcess=Button(root, text="Process", command=codigoBoton,font=("Courier", 20))
botonProcess.config(cursor='hand2')
botonProcess.pack()

root.mainloop()