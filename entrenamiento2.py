import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

### grandCam
import random

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

# Display
from IPython.display import Image, display
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

def get_img_array(img_path, size):
    # img is a PIL image of size 255x255
    
    img = keras.utils.load_img(img_path, (255,255))
    # array is a float32 Numpy array of shape (255, 255, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 255, 255, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, modelo, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        modelo.inputs, [modelo.get_layer(last_conv_layer_name).output, modelo.output]
    )

    print("img_arry: " , img_array)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

   


def gradCam (modelo, tamañoImagen, capa, ruta, nombreFinal):

    model_builder = modelo.summary()
    img_size = (tamañoImagen, tamañoImagen)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions
    
    last_conv_layer_name = capa

    # The local path to our target image
    img_path = ruta

    display(Image(img_path))

    # Prepare image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # Make model
    #model = model_builder(weights="imagenet")

    # Remove last layer's softmax
    modelo.layers[-1].activation = None

    # Print what the top predicted class is
    #preds = model.predict(img_array)
    #print("Predicted:", decode_predictions(preds, top=1)[0])

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, modelo, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    save_and_display_gradcam(img_path, heatmap, nombreFinal)


#################################
ancho=255
alto=255
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
model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding="same", activation="relu", name="conv1"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding="same", activation="relu", name="conv2"))
model.add(MaxPool2D(pool_size=2, strides=2))



#Aplanamiento
model.add(Flatten())
model.add(Dense(256, activation="relu"))


#Capa de salida
model.add(Dense(numeroCategorias, activation="softmax"))


#Traducir de keras a tensorflow
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score(average='weighted')])
model.fit(x=imagenes, y=probabilidades, epochs=15, batch_size=32)

#Prueba del modelo
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1], "Precision=", resultados[2], "Recall=", resultados[3], "F1-Score=", resultados[4])

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


# Guardar modelo
ruta="models/modelo5.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()