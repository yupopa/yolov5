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





uploaded_file = st.file_uploader('', type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)



if uploaded_file is None:
    # Default image.
  image = Image.open("BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg")

    
  model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="best.pt")



  model.results = model(image, size=640)
  model.results.save()


else:
    # User-selected image.
  image = Image.open(uploaded_file)
  model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="best.pt")



  model.results = model(image, size=640)
  model.results.save()
  
## Title.
st.write('# Blood Cell Count Object Detection')


# Convert to JPEG Buffer.
#buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

# Base 64 encode.
#img_str = base64.b64encode(buffered.getvalue())
#img_str = img_str.decode('ascii')


model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="best.pt")



model.results = model(image, size=640)
model.results.save()


# Convert to JPEG Buffer.
#buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')



