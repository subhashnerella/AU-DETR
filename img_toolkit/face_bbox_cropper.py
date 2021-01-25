import cv2
from img_toolkit.face_landmark import FaceLandMark
import config
import numpy as np
import pdb
from collections import defaultdict

class FaceBboxCropper():
    def __init__(self):
        self.landmarker = FaceLandMark()
        
        
    def dlib_face_crop(self,image, landmark_dict):
        h_offset = 50
        w_offset = 20
        sorted_x = np.array(sorted([val[0] for val in landmark_dict.values()]),np.int)
        sorted_y = np.array(sorted([val[1] for val in landmark_dict.values()]),np.int)
        rect = {"top": sorted_y[0] - h_offset, "left": sorted_x[0] - w_offset,
                "width": sorted_x[-1] - sorted_x[0] + 2 * w_offset, "height": sorted_y[-1] - sorted_y[0] + h_offset}
        for key, val in rect.items():
            if val < 0:
                rect[key] = 0
        new_face = image[rect["top"]:rect["top"] + rect["height"], rect["left"]: rect["left"] + rect["width"], ...]
        return new_face, rect
    
    
    
    def get_bboxes(self,orig_img_path,mc_manager=None):
        #pdb.set_trace()
        if mc_manager is not None:
            try:
                image = cv2.imread(config.DATA_ROOT_PATH+orig_img_path)
                result = mc_manager.get(orig_img_path)
                crop_rect = result['crop']
                au_bbox_dict = result['au_bbox_dict']
                cropped_face = image[crop_rect["top"]:crop_rect["top"] + crop_rect["height"],
                               crop_rect["left"]: crop_rect["left"] + crop_rect["width"], ...]
                box_labels = result['box_labels']
                crop_face = cv2.resize(crop_face, config.IMG_SIZE)
                return au_bbox_dict,cropped_face,box_labels
            except Exception:
                pass
        image = cv2.imread(config.DATA_ROOT_PATH+orig_img_path)
        landmarks,_ = self.landmarker.landmark(image)
        
        self.landmarker.update_landmarks(landmarks)
        crop_face,rect = self.dlib_face_crop(image,landmarks)
        crop_face = cv2.resize(crop_face, config.IMG_SIZE)
        top, left, width, height = rect["top"], rect["left"], rect["width"], rect["height"]
        
        
        bbox = {}
        for i,values in config.AU_BOX.items():
            polygon = []
            for value in values:
                polygon.append(landmarks[value])
            polygon = np.array(polygon,np.float32)
            polygon -= np.array([left,top])
            polygon *= np.array([crop_face.shape[1]/width,crop_face.shape[0]/height])
            s = np.amin(polygon,axis=0)
            e= np.amax(polygon,axis=0)
            bbox[i] = np.append(s[np.newaxis,...],e[np.newaxis,...],axis=0)
            
        #au_box_dict,box_labels = self._map_box_au(bbox)
        if mc_manager is not None:
            save_dict= {"crop":rect,'au_bbox_dict':au_box_dict,'box_labels':box_labels}
            mc_manager.set(orig_img_path,save_dict)
        return bbox,crop_face#,box_labels
    
    def _map_box_au(self,bbox_dict):
        au_box_dict = {}
        labels = []
        idx = 1
        for key,values in config.AU_BOX_map.items():
            rect = bbox_dict[key]
            for value in values:
                au_box_dict[idx] = rect
                idx+=1
            labels.extend(values)
                
        return au_box_dict,labels