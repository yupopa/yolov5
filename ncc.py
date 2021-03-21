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




st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)





if uploaded_file is None:
  return None

else:
  # User-selected image.
  image= uploaded_file.read()
  image = Image.open(uploaded_file)


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
results.save()


# Convert to JPEG Buffer.
#buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

# Display image.
st.image(image,
         use_column_width=True)


