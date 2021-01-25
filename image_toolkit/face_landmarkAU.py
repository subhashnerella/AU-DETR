
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
        self.face_dict = OrderedDict({'1':[36,37,38,39,40,41],
                         '2':[42,43,44,45,46,47],
                         '3':[28,31,33,35],
                         '4':[17,18,19,20,21],
                         '5':[22,23,24,25,26],
                         '6':[48,49,52,52,53,54,55,56,57,58,59]         
                        })

        self.class_id = {'eye':1,     
                        'nose':2,
                        'mouth':3,
                        'eyebrow':4
                        }

        
    def __call__(self, img_path,need_txt_img=False):
        
        image = cv2.imread(data_path+img_path)     
        shape = self.landmark(image)      
        new_image = None
        landmarks = {i: [shape.part(i).x, shape.part(i).y] for i in range(0, 68)}
#         if need_txt_img:
#             new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             for i in range(0, 68):
#                 cv2.putText(new_image, str(i), (shape.part(i).x,shape.part(i).y), font, 0.5, (255, 255, 255), 1)
#                 cv2.circle(new_image, (shape.part(i).x, shape.part(i).y),
#                            1, (0, 0, 255), thickness=3)
        rect = self._face_crop(landmarks)
        bboxes = self.gen_bboxes(landmarks)
        new_face = image[rect["top"]:rect["top"] + rect["height"], rect["left"]: rect["left"] + rect["width"], ...]
        #pdb.set_trace()
        new_face = cv2.resize(new_face, config.IMG_SIZE) 
        old_size = image.shape[:2]
        
        def calc_offset(polygon):
            top, left, width, height = rect["top"], rect["left"], rect["width"], rect["height"]
            polygon = polygon.astype(np.float32)
            polygon -= np.array([left, top])   
            h_scale = config.IMG_SIZE[0]/height; w_scale = config.IMG_SIZE[1]/width
            polygon = polygon*[w_scale,h_scale]
            return polygon

        bbox_list = list(map(calc_offset,bboxes.values()))
        bboxes = dict(zip(np.arange(len(bboxes)),bbox_list))
        target = {}
        target['rect'] = rect
        target['bboxes'] = bboxes
        target['labels_names'] = ['eye','eye','nose','eyebrow','eyebrow','mouth']
        target['labels'] = list(map(self.class_id.get,target['labels_names']))
        target['labels'] = np.array(target['labels'])-1
        new_face = cv2.cvtColor(new_face,cv2.COLOR_BGR2RGB) 
        #pdb.set_trace()
        if need_txt_img:
            new_image = new_face.copy()
            for box in target['bboxes'].values():
                box = box.astype(np.int)
                new_image = cv2.rectangle(new_image,tuple(box[0]-5),tuple(box[1]+5),(255, 0, 255),2)
        return new_face,target,new_image
    

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
      
        
        
    def gen_bboxes(self, landmarks):
        bbox = {}
        for i,values in self.face_dict.items():
            polygon = []
            for value in values:
                polygon.append(landmarks[value])
            polygon = np.array(polygon,np.int32)
            s = np.amin(polygon,axis=0)
            e= np.amax(polygon,axis=0)
            bbox[i] = np.append(s[np.newaxis,...],e[np.newaxis,...],axis=0)
        return bbox

    
    def _face_crop(self, landmark_dict):
        h_offset = 50
        w_offset = 20
        sorted_x = sorted([val[0] for val in landmark_dict.values()])
        sorted_y = sorted([val[1] for val in landmark_dict.values()])
        rect = {"top": sorted_y[0] - h_offset, "left": sorted_x[0] - w_offset,
                "width": sorted_x[-1] - sorted_x[0] + 2 * w_offset, "height": sorted_y[-1] - sorted_y[0] + h_offset}
        for key, val in rect.items():
            if val < 0:
                rect[key] = 0
        return rect
    
 
    
if __name__ == "__main__":
    land = Faceannotation()
    face_img_path = "BP4D/data/F001/T1/0000.jpg"
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

