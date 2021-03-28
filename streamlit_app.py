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
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st


url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)

urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "fig-03_5.png"
urlretrieve(urll,filenamee)
st.image(filenamee)
st.write('# KAN HÜCRESİ TESPİTİ')


uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)
    model.results = model(img_array, size=640)
    model.results.save()  # or .show()
    st.image("results/image0.jpg")
  

   

    

    
    liste = []
    liste2=[]
    for i in model.results.xywh:
        for j in i:
            for k in j:
                if k == 2:
                    liste.append(j)
                    
    for i in model.results.xywh:
        for j in i:
            for k in j:
                if k == 1:
                    liste2.append(j)

    st.write("number of wbcs",len(liste),"number of rbcs",len(liste2))
