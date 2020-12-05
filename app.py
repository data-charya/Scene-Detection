import pandas as pd
import numpy as np 
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import PIL
import streamlit as st
from keras.models import load_model
import cv2


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = load_model('https://drive.google.com/file/d/1kAhKqMZeSgqoveifCZfolzhFlOklJu3v/view?usp=sharing')
    return model
model = load_model()
st.title("Ai Scene Detection")
st.title("Upload an image to classify")
file = st.file_uploader("Please upload an image here",type=["jpg","png"])


def import_and_predict(image_data,model):
    tested_image = image.load_img(image_data,target_size=(128,128))
    tested_image = image.img_to_array(tested_image)
    tested_image = np.expand_dims(tested_image,axis = 0)
    output = model.predict(tested_image)
    
    return output

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    class_names=['food','landscape','night','plants','portraits']
    string = "This image is most likely is : "+class_names[np.argmax(predictions)]
    st.success(string)
