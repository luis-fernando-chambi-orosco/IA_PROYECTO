import sys
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
# Verificar si el error específico ha ocurrido
try:
    # Tu línea que genera el error
    isinstance(None, tf.distribute.input_lib.DistributedDatasetInterface)
except AttributeError as e:
    if "module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface'" in str(e):
        # Remplazar la clase que causa el error con la alternativa
        from tensorflow.python.distribute import input_lib
        input_lib.DistributedDatasetInterface = input_lib.DistributedDatasetSpec
    else:
        # Si el error es diferente, imprimirlo para su análisis
        print(e)
        
K.clear_session()

data_entrenamiento='./data/entrenamiento'
data_validacion='./data/validacion'

#parametros

epocas=20
altura, longitud =100,100
batch_size=32
pasos=100
pasos_validacion=100
filtrosconv1=32
filtrosconv2=64
tamaño_filtro1=(3,3)
tamaño_filtro2=(2,2)
tamaño_pool=(2,2)
clases=2
lr=0.0005

#pre procesamiento de imagenes
entrenamiento_datagen=ImageDataGenerator(
	rescale=1./255,
	shear_range=0.3,
	zoom_range=0.3,
	horizontal_flip=True
)
validacion_datagen=ImageDataGenerator(
	rescale=1./255
)
imagen_entrenamiento=entrenamiento_datagen.flow_from_directory(
	data_entrenamiento,
 target_size=(altura,longitud),
 batch_size=batch_size,
 class_mode='categorical'
)
imagen_validacion=validacion_datagen.flow_from_directory(
	data_validacion,
	target_size=(altura,longitud),
	batch_size=batch_size,
 	class_mode='categorical'
)
# crear la red CNN
cnn=Sequential()
cnn.add(Convolution2D(filtrosconv1,tamaño_filtro1,padding='same',input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))
cnn.add(Convolution2D(filtrosconv2,tamaño_filtro2,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))

cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation='softmax'))

cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'])

#cnn.compile(loss='categorical_crossentropy',metrics=['accuracy'])
cnn.fit(imagen_entrenamiento,
        epochs=epocas,
        steps_per_epoch=pasos,
        validation_data=imagen_validacion,
        validation_steps=pasos_validacion)

dir='./modelo/'

if not os.path.exists(dir):
         os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')    