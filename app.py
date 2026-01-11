import streamlit as st
import numpy as np
from PIL import Image
import joblib


# loading the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model,scaler

model,scaler = load_model()


# image preprocessing
def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((64,64))
    image = np.array(image).flatten().reshape(1,-1)
    image = scaler.transform(image)
    return image



st.title("Smile Stalker - Classifying smile and non-smile images")

upload_file = st.file_uploader("Upload an Image",type=["jpg","jpeg","png"])


if upload_file is not None:
    # open image using PIL
    img = Image.open(upload_file)
    st.image(img,caption="Uploaded Image")



    # preprocess and make predictions
    pre_img = preprocess_image(img)
    prediction_proba = model.predict_proba(pre_img)[0][1]
    prediction = model.predict(pre_img)


    # Display result
    smile_score = int(prediction_proba*100)
    st.slider("Smile Sccore",0,100,smile_score,disabled=True)


    if prediction == 1:
        st.success("The person in the image is smiling")
    
    else:
        st.warning("The person in the image is not smiling")