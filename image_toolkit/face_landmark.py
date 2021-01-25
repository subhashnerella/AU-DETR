
import cv2
import dlib
import numpy as np
from functools import lru_cache
from matplotlib import pyplot as plt
import pdb
from PIL import Image
from collections import OrderedDict

face_dict = OrderedDict({'1':[36,37,38,39,40,41],
                         '2':[42,43,44,45,46,47],
                         '3':[27,31,33,35],
                         '4':[17,18,19,20,21],
                         '5':[22,23,24,25,26],
                         '6':[48,49,52,52,53,54,55,56,57,58,59]         
                        })

face_classes = {'eye':['1','2'],     
            'nose':['3'],
            'mouth':['6'],
            'eyebrow':['4','5']
            }

class FaceLandMark():

    def __init__(self, model_file_path):
        self.predictor = dlib.shape_predictor(model_file_path)
        self.detector = dlib.get_frontal_face_detector()
        print("FaceLandMark init call! {}".format(model_file_path))

    def landmark_from_path(self, face_file_path):
        image = cv2.imread(face_file_path,cv2.IMREAD_GRAYSCALE)
        return self.landmark(image)


    def landmark(self, image, need_txt_img=False):
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = image
        if image is None:
            print("None image!")
        if image.ndim >= 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # clahe_image = clahe.apply(gray)
        small_gray_img = cv2.resize(gray, (int(gray.shape[1] * 1/4.0), int(gray.shape[0] * 1/4.0)))
        try:
            dets = self.detector(small_gray_img, 0)  # boost speed for small image, detect bounding box of face , slow legacy
            # only one face,so the dets will always be length = 1
            small_d = dets[0]
            # dlib.dlib.rectangle(d.left(),d.top(),d.right(),d.bottom())
            d = dlib.rectangle(small_d.left() * 4, small_d.top() * 4, small_d.right()* 4, small_d.bottom() * 4)
        except IndexError:
            dets = self.detector(gray, 0)
            d = dets[0]
        shape = self.predictor(gray, d)
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_image = None
        if need_txt_img:
            new_image = image.copy()
            for i in range(0, 68):
                cv2.putText(new_image, str(i), (shape.part(i).x,shape.part(i).y), font, 0.5, (255, 255, 255), 1)
                cv2.circle(new_image, (shape.part(i).x, shape.part(i).y),
                           1, (0, 0, 255), thickness=3)
        return {i: [shape.part(i).x, shape.part(i).y]
                    for i in range(0, 68)}, image, new_image
    


        
        
    def gen_bboxes(self, landmark):
        bbox = {}
        for i,values in face_dict.items():
            polygon = []
            for value in values:
                polygon.append(landmarks[value])
            polygon = np.array(polygon,np.int32)
            s = np.amin(polygon,axis=0)
            e= np.amax(polygon,axis=0)
            bbox[i] = np.np.append(s,e-s)
        return bbox


if __name__ == "__main__":
    land = FaceLandMark('./shape_predictor_68_face_landmarks.dat')
    face_img_path = "/data/datasets/users/subhash/BP4D/data/F001/T1/0000.jpg"
    trn_img = cv2.imread(face_img_path, cv2.IMREAD_COLOR)
    img= Image.open(face_img_path)
    landmarks, _ , _= land.landmark(image=trn_img)
    fig = plt.figure(figsize=(10,20))
    plt.imshow(img)
    for i,(x,y) in enumerate(landmarks.values()):
        plt.scatter(x,y,c='b')

        plt.annotate(i,(x,y),c='w')
    plt.axis('off')
    plt.savefig('test.png')

