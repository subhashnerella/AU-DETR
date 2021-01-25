
import cv2
import dlib
import numpy as np
from functools import lru_cache
from matplotlib import pyplot as plt
import pdb
from PIL import Image
from collections import OrderedDict
import config

data_path = config.DATA_ROOT_PATH
dlib_model = config.DLIB_MODEL_PATH


class Faceannotation():

    def __init__(self, mc_manager=None, model_file_path = dlib_model):
        self.predictor = dlib.shape_predictor(model_file_path)
        self.detector = dlib.get_frontal_face_detector()
        self.mc_manager = mc_manager
        
    def landmark(self, image):
        gray = image
        if image is None:
            print("None image!")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small_gray_img = cv2.resize(gray, (int(gray.shape[1] * 1/4.0), int(gray.shape[0] * 1/4.0)))
        try:
            dets = self.detector(small_gray_img, 0) 
            small_d = dets[0]
            # dlib.dlib.rectangle(d.left(),d.top(),d.right(),d.bottom())
            d = dlib.rectangle(small_d.left() * 4, small_d.top() * 4, small_d.right()* 4, small_d.bottom() * 4)
        except IndexError:
            dets = self.detector(gray, 0)
            d = dets[0]
        shape = self.predictor(gray, d)
        return shape   
    
    def __call__(self,img_path,need_txt_img=False):
        image = cv2.imread(data_path+img_path)     
        shape = self.landmark(image)      