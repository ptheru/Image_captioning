import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from google.cloud import storage
from PIL import Image
import time


# Setup environment credentials 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_app_cred.json" 
PROJECT = "caption-gen-cloud-func" 
GC_URL = "https://[REGION]-[PROJECTID].cloudfunctions.net/[CLOUD FUCNTION]"

storage_client = storage.Client(project=PROJECT)
bucket = storage_client.get_bucket('caption_gen_ptheru')

# ### Streamlit code ###
st.title("Welcome to Caption Generator 📸 🗣")
st.header("Caption generator describes what's in your image!")


# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image",
                                 type=["jpg"])


session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:   
    session_state.uploaded_image = uploaded_file.read() 
    filename = '/streamlit_images/' + str(time.strftime("%Y%m%d%H%M%S")) + '.jpg'
    blob = bucket.blob(filename)

    param = {"image": filename}

    image = Image.open(uploaded_file)
    image.save('test_file.jpg')

    blob.upload_from_filename('test_file.jpg')
  
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Generate Caption")


def update_logger(image, user_caption=None):
    """
    Function for tracking feedback given in app, updates and returns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "user_caption": user_caption
    }   
    return logger

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
 
    r = requests.post(GC_URL, json=param)
    session_state.decoded_caption = r.content
    st.write(f"Caption: {session_state.decoded_caption}")  

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=filename))
    elif session_state.feedback == "No":
        session_state.decoded_caption = st.text_input("What should the correct caption be?")
        if session_state.decoded_caption:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=filename,
                                user_caption=session_state.decoded_caption))

os.remove('test_file.jpg')
