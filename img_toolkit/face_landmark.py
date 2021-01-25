
import cv2
import dlib
import numpy as np
from .geometry_utils import sort_clockwise
import config
import pdb

dlib_model = config.DLIB_MODEL_PATH

class FaceLandMark():

    def __init__(self, model_file_path=dlib_model):
        self.predictor = dlib.shape_predictor(model_file_path)
        self.detector = dlib.get_frontal_face_detector()
        print("FaceLandMark init call! {}".format(model_file_path))


    def landmark(self, image, need_txt_img=False):
        gray = image
        if image is None:
            print("None image!")
        if image.ndim >= 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #small_gray_img = cv2.resize(gray, (int(gray.shape[1] * 1/4.0), int(gray.shape[0] * 1/4.0)))
#         try:
#             dets = self.detector(small_gray_img, 0) 
#             small_d = dets[0]
#             # dlib.dlib.rectangle(d.left(),d.top(),d.right(),d.bottom())
#             d = dlib.rectangle(small_d.left() * 4, small_d.top() * 4, small_d.right()* 4, small_d.bottom() * 4)
#         except IndexError:
        dets = self.detector(gray, 0)
        if len(dets)>=1:
            d = dets[0]
        else:
            return 0,0            
        shape = self.predictor(gray, d)
        return {str(i): [shape.part(i).x, shape.part(i).y] for i in range(0, 68)}, image
    

    def split_ROI(self, landmark):
        def trans_landmark2pointarr(landmark_ls):
            point_arr = []
            for land in landmark_ls:
                if land.endswith("uu"):
                    land = int(land[:-2])
                    x, y = landmark[land]
                    y -= 40
                    point_arr.append((x, y))
                elif land.endswith("u"):
                    land = int(land[:-1])
                    x, y = landmark[land]
                    y -= 20
                    point_arr.append((x, y))
                elif "~" in land:
                    land_a, land_b = land.split("~")
                    land_a = int(land_a)
                    land_b = int(land_b)
                    x = (landmark[land_a][0] + landmark[land_b][0]) / 2
                    y = (landmark[land_a][1] + landmark[land_b][1]) / 2
                    point_arr.append((x, y))
                else:
                    x, y = landmark[int(land)]
                    point_arr.append((x, y))
            return sort_clockwise(point_arr)

        polygons = {}
        for roi_no, landmark_ls in config.ROI_LANDMARK.items():
            polygon_arr = trans_landmark2pointarr(landmark_ls)
            polygon_arr = polygon_arr.astype(np.int32)
            polygons[int(roi_no)] = polygon_arr
        return polygons
    
    
    def update_landmarks(self,landmarkdict):
        try:
            offset = 50
            for landmark_list in list(config.AU_BOX.values()):
                for landmark in landmark_list:
                    if 'u' in landmark:
                        key = landmark[:-1]
                        x, y = landmarkdict[key]
                        y -= offset
                        landmarkdict[landmark] = [x,y]
                    elif '~' in landmark:
                        key1,key2 = landmark.split('~')
                        x1,y1 = landmarkdict[key1];x2,y2 = landmarkdict[key2];
                        landmarkdict[landmark] = [(x1+x2)/2,(y1+y2)/2]
                    else:
                        continue
        except:
            pdb.set_trace()
        
        


