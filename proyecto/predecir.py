import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Tamaño de la imagen
longitud, altura = 150, 150

# Variables de ruta donde se encuentra el modelo
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

# Cargar el modelo y los pesos
cnn = load_model(modelo)
cnn.load_weights(pesos)

# Función para la predicción
def predecir_imagen(ruta):
    x = load_img(ruta, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    
    if respuesta == 0:
        return 'Auto'
    elif respuesta == 1:
        return 'Moto'

# Función para cargar y predecir una imagen seleccionada por el usuario
def seleccionar_imagen():
    ruta_imagen = filedialog.askopenfilename(title='Seleccionar Imagen', filetypes=[('Archivos de Imagen', '*.png;*.jpg;*.jpeg')])
    
    if ruta_imagen:
        # Mostrar la imagen en la interfaz
        img = Image.open(ruta_imagen)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel_imagen.config(image=img)
        panel_imagen.image = img
        
        # Realizar la predicción
        resultado_prediccion = predecir_imagen(ruta_imagen)
        etiqueta_resultado.config(text=f'Resultado de la Predicción: {resultado_prediccion}')

# Crear la interfaz
app = tk.Tk()
app.title('Clasificador de Autos y Motos')

# Botón para seleccionar una imagen
btn_seleccionar = tk.Button(app, text='Seleccionar Imagen', command=seleccionar_imagen)
btn_seleccionar.pack(pady=10)

# Panel para mostrar la imagen seleccionada
panel_imagen = tk.Label(app)
panel_imagen.pack()

# Etiqueta para mostrar el resultado de la predicción
etiqueta_resultado = tk.Label(app, text='Resultado de la Predicción: ')
etiqueta_resultado.pack(pady=10)

# Iniciar la interfaz
app.mainloop()
