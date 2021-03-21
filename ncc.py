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
    
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)
with torch.no_grad():
    output=model(image)
output0= output[0]
st.write(print(output0))

    
    
r = requests.get(uploaded_file, stream=True)
image = Image.open(io.BytesIO(r.content))


    
    
    
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



