# Utils for preprocessing data etc 
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
from model import get_cnn_model,TransformerEncoderBlock,TransformerDecoderBlock,ImageCaptioningModel,compile_model

SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
MODEL_PATH = '/tmp/checkpoint.ckpt'


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
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels=3)
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


