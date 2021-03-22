import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy
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
    img_array = np.array(image)







    


    
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)


model.results = model(img_array, size=640)


model.results.save()  # or .show()
st.image("results/image0.jpg")

   






