import torch
import streamlit as st
import requests
import glob
import torch
from urllib.request import urlretrieve
from PIL import Image
import streamlit as st
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

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='best.pt')

from PIL import Image

uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)
if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(uploaded_file)
    
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')   

  



# Inference
results = model(image, size=640)  # includes NMS
results.print()  
results.save()  # or .show()
st.image("results/image",
         use_column_width=True)

# Data
