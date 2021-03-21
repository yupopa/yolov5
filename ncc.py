import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import torch
from urllib.request import urlretrieve
from fastai.vision.widgets import *
from fastai.vision.all import *
import cv2


url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)



uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = torch.tensor(np.array(image)).permute((2,0,1)).unsqueeze(0)
    image = image.float()/255
    
    
    
    

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from matplotlib import pyplot as plt
import numpy as np

def load_file_and_process(path):
    image = load_img(bytes.decode(path.numpy()), target_size=(224, 224))
    image = img_to_array(image)
    image = tf.image.central_crop(image, np.random.uniform(0.50, 1.00))
    return imagee

train_dataset = tf.data.Dataset.list_files(image)
train_dataset = train_dataset.map(lambda x: tf.py_function(load_file_and_process, [x], [tf.float32]))

for f in train_dataset:
  for l in f:
    image = np.array(array_to_img(l))
    plt.imshow(imagee)
    
    

    
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)
with torch.no_grad():
    output=model(image)
output0= output[0]
st.write(print(output0))



    
    
    
    #try the below line instead of Image.open()
  # This portion is part of my test code
   # byteImgIO = io.BytesIO()
    #byteImg = Image.open(uploaded_file)
    #byteImg.save(byteImgIO, "PNG")
   # byteImgIO.seek(0)
   # byteImg = byteImgIO.read()


# Non test code
   # dataBytesIO = io.BytesIO(byteImg)
   # Image.open(dataBytesIO)
   

# Convert to JPEG Buffer.
#buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')
#img_str = base64.b64encode(buffered.getvalue())
#img_str = img_str.decode('ascii')


st.write('# Blood Cell Count Object Detection')



model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)
model.autoshape()


model.results = model(image)
st.image(image)



