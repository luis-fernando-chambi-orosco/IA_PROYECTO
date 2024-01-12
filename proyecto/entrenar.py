import sys
import os
## usamos keras dentro de tensorflow
from keras.preprocessing.image import ImageDataGenerator## librerias de porcesador de imagenes
from tensorflow.python.keras.models import Sequential#esta libreria nos permite realizar redes neuronales secuenciales 
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D# se reliza las capas de convolucion 
from tensorflow.python.keras import backend as K #ayuda a finalizar cualquiera sesion activa para iniciar el procesos desde cero 

#finalizamos cuzlquier sesison de keras que este activa en la maquina 
K.clear_session()


#variables de las rutas en cual se encuentra las imagenes a entrenar 
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

#parametros
epocas=20
# tamanño de procesaiment de imagenes
longitud, altura = 150, 150
#numero de imagenes que se envia al modelo en cada uno de los pasos 
batch_size = 32
#numero de veces que se a procesar la informacion  en ada una de las epocas 
pasos = 500
validation_steps = 300
#profundida de imagen 
filtrosConv1 = 32#primera convolucion
filtrosConv2 = 64#segunda convolucion 
#tamaño de filtros
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)#tamaño del filtro de maxpooling
clases = 2#auto, moto
lr = 0.0004# tamaño de ajuste de la red neuronal para acercarse a la una solucion optima 


##Preparamos nuestras imagenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,#reescalado de pixeles para hacer mas eficientes en entrenamiento
    shear_range=0.2,#permite inclinar las imagenes porque no siempre vana esatr parados 
    zoom_range=0.2,#permite hacer zoom cuando un peroor o gato este incompleto
    horizontal_flip=True)#permite que la red neuronal aprensa direccionalida la red neuronal 

# reescalamaiento de imagenes para validacion las imagenes tal cual  son o sea sin  modificarlas 
test_datagen = ImageDataGenerator(rescale=1. / 255)

#se leeran imagenes del directorio de entrenamientos y se procesa las iamgnes de acuerod a los parametros
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')#clasificacion por categoria (gatos y perros )

#se leeran imagenes del directorio de validacion y se procesa las imagenes de acuerdo a los parametros
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')#clasificacion por categoria (gatos y perros )

#crear la red convulocional CNN
cnn = Sequential() #varaible de red convolucional se crean las capas en secuencia 
# se agraga la primera capa cambiando el tamaño de las imagenes
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
# se agrega una capa pooling
cnn.add(MaxPooling2D(pool_size=tamano_pool))
# se agraga la primera capa cambiando el tamaño de las imagenes
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
# se agrega una capa pooling
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# se procede a empezar la clasificacion 
cnn.add(Flatten())# se hacen las imagenes planas 
cnn.add(Dense(256, activation='relu'))#interconexion de neuronas y vamos a tener una dimesion de 256
cnn.add(Dropout(0.5))# se apaga el 50% de las neuronas para evitar que busque caminos mas cortos al clasficar la  data 
cnn.add(Dense(clases, activation='softmax'))# ultima capas de dos neuronas en el cual sofmax nos ayudara a decir si la imagen es perro po gato  de acuerdo a a clasificacion correcta 

# el algoritmo puede ver que tan bien o tan mal  va de acuerdo a la perdida 
cnn.compile(loss='categorical_crossentropy',
            metrics=['accuracy'])



#se procede a entrenar el modelo de acuerdo a los parametros 
cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

#proceso d guardar el modelo en u archivo llamado  modelo sino lo tien lo crea 
target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
  #guardamos en archivo la estructura y pesos de nuestro modelo 
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')