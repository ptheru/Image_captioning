import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from utils import load_model,load_and_prep_image,index_to_word, output_captions,update_logger

# Setup environment credentials 

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_app_creds.json" 
# PROJECT = "caption-generator-priya" 
# REGION = "us-central1" # GCP region (where the model is hosted)

# ### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Caption Generator 📸 🗣")
st.header("Caption generator describes what's in your image!")

SEQ_LENGTH = 25
max_decoded_sentence_length = SEQ_LENGTH - 1

#locally
# with open('../caption_tokenizer.pkl','rb') as f:
#     tokenizer = pickle.load(f)

#docker location
with open('caption_model/caption_tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)
    
index_to_word_dict = index_to_word(tokenizer)

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image):
    """
    Takes an image and uses model (a trained TensorFlow model) to generate a
    caption.

    Returns:
     image (preproccessed)
     decoded_caption (caption)
    """
    image = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)

    caption_model = load_model()

    image = caption_model.cnn_model(image)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(image, training=False)

    # # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        _tokenized_caption = output_captions(tokenizer,[decoded_caption])[:,:-1]
        mask = tf.math.not_equal(_tokenized_caption, 0) #helps with regular padding mask
        predictions = caption_model.decoder(
            _tokenized_caption, encoded_img, training=False, mask=mask 
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_to_word_dict[sampled_token_index]
        #print(sampled_token)
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    
    return image,decoded_caption

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image",
                                 type=["jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Generate Caption")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.decoded_caption = make_prediction(session_state.uploaded_image)
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
        print(update_logger(image=session_state.image))
    elif session_state.feedback == "No":
        session_state.decoded_caption = st.text_input("What should the correct caption be?")
        if session_state.decoded_caption:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                user_caption=session_state.decoded_caption))
