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
    #try the below line instead of Image.open()
    image= cv2.imread("uploaded_file")

    st.image(image, caption='Uploaded Image.')
   

# Convert to JPEG Buffer.
#buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')
#img_str = base64.b64encode(buffered.getvalue())
#img_str = img_str.decode('ascii')


st.write('# Blood Cell Count Object Detection')



model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="best.pt")


model.results = model(image)



