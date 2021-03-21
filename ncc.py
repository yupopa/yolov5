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


url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)




uploaded_file = st.file_uploader('', type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)


if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(uploaded_file)
## Title.
st.write('# Blood Cell Count Object Detection')

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="best.pt")

model.results = model(image, size=640)
model.results.save()



