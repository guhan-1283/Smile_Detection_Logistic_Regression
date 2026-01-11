import streamlit as st
import numpy as np
from PIL import Image
import pickle

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model()

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((64, 64))
    image = np.array(image).flatten().reshape(1, -1)
    image = scaler.transform(image)
    return image

st.title("Smile Stalker")
st.subheader("Classifying Smile and Non-Smile Images")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    pre_img = preprocess_image(img)
    prediction_proba = model.predict_proba(pre_img)[0][1]
    prediction = model.predict(pre_img)[0]

    smile_score = int(prediction_proba * 100)

    st.slider("Smile Score", 0, 100, smile_score, disabled=True)

    if prediction == 1:
        st.success("😄 The person in the image is smiling!")
    else:
        st.warning("😐 The person in the image is not smiling.")
