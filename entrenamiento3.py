import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten

import matplotlib as mpl
import matplotlib.pyplot as plt




##################################

def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=rutaOrigen+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

#################################


#################################
ancho=256
alto=256
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=4

cantidaDatosEntrenamiento=[1321, 1339, 1595, 1457]
cantidaDatosPruebas=[300,306,405,300]

#Cargar las imágenes
imagenes, probabilidades=cargarDatos("dataset/train/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)

model=Sequential()
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))

#Capas Ocultas
#Capas convolucionales
model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding="same", activation="relu", name="conv_1"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding="same", activation="relu", name="conv_2"))
model.add(MaxPool2D(pool_size=2, strides=2))


#Aplanamiento
model.add(Flatten())
model.add(Dense(256, activation="relu"))



#Capa de salida
model.add(Dense(numeroCategorias, activation="softmax"))


#Traducir de keras a tensorflow
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score(average='weighted')])
model.fit(x=imagenes, y=probabilidades, epochs=5, batch_size=60)



#Prueba del modelo
imagenesPrueba, probabilidadesPrueba = cargarDatos("dataset/test/", numeroCategorias, cantidaDatosPruebas, ancho, alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Accuracy=", resultados[1], "Precision=", resultados[2], "Recall=", resultados[3], "F1-Score=", resultados[4])

predicciones_prueba = model.predict(imagenesPrueba)
#######################################################################

# Obtén las predicciones del modelo
predicciones_prueba = model.predict(imagenesPrueba)

# Convierte las predicciones y los valores reales a etiquetas simples
predicciones_prueba_etiquetas = np.argmax(predicciones_prueba, axis=1)
probabilidades_prueba_etiquetas = np.argmax(probabilidadesPrueba, axis=1)

# Calcula la matriz de confusión
matriz_confusion = confusion_matrix(probabilidades_prueba_etiquetas, predicciones_prueba_etiquetas)
print("Matriz de confusión:")
print(matriz_confusion)

# Visualiza la matriz de confusión
plt.figure(figsize=(10,7))
plt.imshow(matriz_confusion, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Matriz de Confusión')
plt.xlabel('Clases Predichas')
plt.ylabel('Clases Verdaderas')
plt.show()

#######################################################################


# Guardar modelo
ruta="models/modelo3.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()
