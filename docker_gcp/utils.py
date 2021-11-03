# Utils for preprocessing data etc 
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
from model import get_cnn_model,TransformerEncoderBlock,TransformerDecoderBlock,ImageCaptioningModel,compile_model

SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
MODEL_PATH = 'caption_model/checkpoint.ckpt'
#MODEL_PATH = '/Users/ypxt035/Desktop/Priya/ml_deployment_tutorial/Image-captioning-exp/caption_model/checkpoint.ckpt'


# def predict_json(project, region, model, instances, version=None):
#     """Send json data to a deployed model for prediction.

#     Args:
#         project (str): project where the Cloud ML Engine Model is deployed.
#         model (str): model name.
#         instances ([Mapping[str: Any]]): Keys should be the names of Tensors
#             your deployed model expects as inputs. Values should be datatypes
#             convertible to Tensors, or (potentially nested) lists of datatypes
#             convertible to Tensors.
#         version (str): version of the model to target.
#     Returns:
#         Mapping[str: any]: dictionary of prediction results defined by the 
#             model.
#     """
#     # Create the ML Engine service object
#     prefix = "{}-ml".format(region) if region else "ml"
#     api_endpoint = "https://{}.googleapis.com".format(prefix)
#     client_options = ClientOptions(api_endpoint=api_endpoint)

#     # Setup model path
#     model_path = "projects/{}/models/{}".format(project, model)
#     if version is not None:
#         model_path += "/versions/{}".format(version)

#     # Create ML engine resource endpoint and input data
#     ml_resource = googleapiclient.discovery.build(
#         "ml", "v1", cache_discovery=False, client_options=client_options).projects()
#     instances_list = instances.numpy().tolist() # turn input into list (ML Engine wants JSON)
    
#     input_data_json = {"signature_name": "serving_default",
#                        "instances": instances_list} 

#     request = ml_resource.predict(name=model_path, body=input_data_json)
#     response = request.execute()
    
#     # # ALT: Create model api
#     # model_api = api_endpoint + model_path + ":predict"
#     # headers = {"Authorization": "Bearer " + token}
#     # response = requests.post(model_api, json=input_data_json, headers=headers)

#     if "error" in response:
#         raise RuntimeError(response["error"])

#     return response["predictions"]

def load_model():
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder)

    caption_model = compile_model(caption_model)

    #Load model weights
    caption_model.load_weights(MODEL_PATH)
    return caption_model

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=299):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (299, 299, 3).
  """
  #image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(filename, channels=3)
  image = tf.image.resize_with_pad(image, img_shape, img_shape)
  image = tf.image.convert_image_dtype(image, tf.float32)
  #image = preprocess_input(image)
  return image

def output_captions(tokenizer,dta):
  train_seqs = tokenizer.texts_to_sequences(dta)

  # Pad each vector to the max_length of the captions
  # If you do not provide a max_length value, pad_sequences calculates it automatically
  tokenized_captions = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post',maxlen=SEQ_LENGTH)
  return tokenized_captions

def index_to_word(tokenizer):
  return dict(zip(tokenizer.word_index.values(),tokenizer.word_index.keys()))


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
