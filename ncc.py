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


##########
##### Set up sidebar.
##########

# Add in location to select image.

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)





##########
##### Set up main app.
##########

## Title.
st.write('# Blood Cell Count Object Detection')


import streamlit as st
from PIL import Image
import numpy as np

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

image = Image.open(img_file_buffer)
img_array = np.array(image)

if image is not None:
    st.image(
        image,
        caption=f"You amazing image has shape {img_array.shape[0:2]}",
        use_column_width=True,
    )


# Convert to JPEG Buffer.
#buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

base64_decoded = base64.b64decode(test_image_base64_encoded)

image = Image.open(io.BytesIO(base64_decoded))
image_np = np.array(image)
image_torch = torch.tensor(np.array(image))
# Base 64 encode.
#img_str = base64.b64encode(buffered.getvalue())
#img_str = img_str.decode('ascii')


model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="best.pt")



model.results = model(image_torch, size=640)
results.save()


# Convert to JPEG Buffer.
#buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

# Display image.
st.image(image,
         use_column_width=True)


