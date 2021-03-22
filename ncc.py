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
    image = image.open(uploaded_file)

    image = Image.Image.load(uploaded_file)
    
    
    
    
    


    
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)




    
    
    
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
#data = numpy.array(image)
#im = Image.open(image)
#im = im.convert("1")

#pixels = im.getdata() # returns 1D list of pixels
#n = len(pixels)
#data = numpy.reshape(pixels, im.size) # turn into 2D numpy array

#for row in data:
    # do your processing
    #pass

# Check that the numpy array's data is good
#im2 = Image.new("1", im.size)
#im2.putdata(numpy.reshape(data, [n, 1]))
#im2.show()

model.results = model(image)




