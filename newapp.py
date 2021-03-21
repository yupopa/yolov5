import glob
import torch
from urllib.request import urlretrieve
from PIL import Image
from IPython.display import Image, display
import streamlit as st
from fastai import *
from fastai.vision.widgets import *
from fastai.vision.all import *

filename = "best.pt"




class Predict:
    def __init__(self, filename):
        self.learn_inference = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
 
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])

        if uploaded_file is not None:
            return  PILImage.create((uploaded_file))


        return None
        
        
        

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')
        results = model(img, size=160)  # includes NMS
        results.print()
        results.save()

if __name__=='__main__':
    predictor = Predict(filename)
    
    
