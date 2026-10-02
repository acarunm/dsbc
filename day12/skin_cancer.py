import streamlit as st  
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

model=load_model('skin_cancer.h5')
def process_image(img)
    img=cv2.imread(str(img))
    img=cv2.resize(img,(170,170))
    img=img/255.0 
    img=np.expand_dims(img,axis=0)
    return img

st.title('Deri Kanser resmi sınıflandırma :cancer:')
st.write('Resim seç, model kanser olup olmadığını tahmin etsin!')

file=st.file_uploader('Bir resim yükle',type=['jpg','jpeg','png'])

if file is not None: # Resim yuklenmisse burasi calisacak
    img=Image.open(file)
    st.image(img, caption='Yuklenen Resim')
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)
    class_names=['Kanser değil','Kanser']
    st.write(class_names[predicted_class])