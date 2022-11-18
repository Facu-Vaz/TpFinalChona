#Importar dependencias
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

 
@st.cache(allow_output_mutation=True)

#Cargar modelo 
def load_model():
  model=tf.keras.models.load_model('chona.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

#Mostrar texto
st.write("""
         # Image Classification
         """
         )

#Mostrar texto
st.write('### Draw a digit in 0-9 in the box below')

#Barra para variar el grosor del pincel
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)

#Botono boolean para decidir si se conrre la prediccion automaticamente o no
realtime_update = st.sidebar.checkbox("Update in realtime", True)
 
#Crear componente de canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=400,
    width=400,
    drawing_mode='freedraw',
    key="canvas",
)

#Se corre si escribiste algo
if canvas_result.image_data is not None:
     
    #Obetener la array del imagen
    input_numpy_array = np.array(canvas_result.image_data)
     
     
    #Convertir a imgen de PIL y guardar
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image.save('user_input.png')
     
    #Convertir a escala de grises
    input_image_gs = input_image.convert('L')
    
    #Resizear a 28x28
    input_image_gs_resize = input_image_gs.resize((28,28), Image.ANTIALIAS)

    #Convertir a np array
    input_image_gs_resize_np = np.array(input_image_gs_resize)

    #Obtener y escribir la prediccion
    pred = model.predict(input_image_gs_resize_np.reshape(1,28,28,1))
    st.write('### The digit is classified as', np.argmax(pred))
    
 
    
