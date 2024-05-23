import tflite_runtime.interpreter as tflite
import numpy as np
from keras_image_helper import create_preprocessor
from PIL import Image
from io import BytesIO
from urllib import request

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img
def prepare_image(img, target_size = (150, 150)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    x = np.array(img, dtype='float32')
    X = np.array([x])
    return X
def preprocess_input(X):
    X /= 255
    return X

classes = ["dino", "dragon"]


def predict(url):
    img = download_image(url)
    x = prepare_image(img, target_size = (150, 150))
    X = preprocess_input(x)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return dict(zip(classes, preds[0]))

def lambda_handler(event, context):
    url = event['url']
    preds = predict(url)
    return {
        'StatusCode':200,
        "body":str(preds)
    }


