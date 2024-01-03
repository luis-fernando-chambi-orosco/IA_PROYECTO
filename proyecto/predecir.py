import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Tamaño de la imagen
longitud, altura = 150, 150

# Variables de ruta donde se encuentra el modelo
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

# Cargar el modelo y los pesos
cnn = load_model(modelo)
cnn.load_weights(pesos)

# Función para la predicción
def predict(file):
    x = load_img(file, target_size=(longitud, altura))  # Variable para cargar la imagen
    x = img_to_array(x)  # La imagen se convierte en arreglo
    x = np.expand_dims(x, axis=0)  # Se añade una dimensión extra en el eje cero para procesar la información sin problemas

    arreglo = cnn.predict(x)  # Se procede a ejecutar la predicción, que devuelve un arreglo en dos dimensiones
    resultado = arreglo[0]  # Se recupera el arreglo de la dimensión cero

    respuesta = np.argmax(resultado)  # Se recupera el valor de índice más alto

    # Se crea una condición if para clasificar como gato o perro
    if respuesta == 0:
        print('Gato')
    elif respuesta == 1:
        print('Perro')
    return respuesta

# Ejemplo de uso con una foto de otro animal
# predict('gatodeprueba.jpg')
predict('C:/Users/Aimer/Downloads/perrodeprueba.jpg')

        