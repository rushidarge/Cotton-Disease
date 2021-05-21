import streamlit as st
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

st.write("""
         # Cotton Disease Prediction
         """
         )
st.write("This is a simple Cotton Disease Prediction web app to predict your disease of cotton plants")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])	

# load weights into new model
model = load_model("prediction_model_ocr.h5")

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img_array = np.array(image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict('out.jpg')
    if result[0][0] == 1:
        prediction = 'fresh cotton leaf'
    else:
        prediction = 'diseased cotton leaf' 
    st.write(prediction)
