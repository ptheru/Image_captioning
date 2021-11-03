import os
import json
import requests
import tensorflow as tf
import numpy as np
import pickle
from utils import load_model,load_and_prep_image,index_to_word, output_captions,update_logger
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.get_bucket('caption_gen_ptheru')

def download(filename):
    blob_weights = bucket.blob(filename)   
    blob_weights.download_to_filename('/tmp/' + filename.split('/')[-1])

SEQ_LENGTH = 25
max_decoded_sentence_length = SEQ_LENGTH - 1
caption_model = None

def generate_caption(request):
    """
    Takes an image and uses model (a trained TensorFlow model) to generate a
    caption.

    Returns:
     image (preproccessed)
     decoded_caption (caption)
    """
    global caption_model,tokenizer,index_to_word_dict
    
    request = request.get_json()
    image = request['image']

    download(image)

    image = image.split('/')[-1]
    image = load_and_prep_image('/tmp/' + image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)

    if not caption_model:
        download('checkpoint.ckpt.data-00000-of-00001')
        download('checkpoint.ckpt.index')
        download('checkpoint')
        download('caption_tokenizer.pkl')
        tokenizer = pickle.load(open('/tmp/caption_tokenizer.pkl', "rb"))
        index_to_word_dict = index_to_word(tokenizer)
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
    
    return decoded_caption

