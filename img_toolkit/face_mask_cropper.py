import cv2
import functools
from img_toolkit.face_landmark import FaceLandMark
import config
import numpy as np
from collections import defaultdict
import json
import ast 
import pdb

class FaceMaskCropper(object):
    landmark = FaceLandMark()
    
    @staticmethod
    def dlib_face_crop(image, landmark_dict):
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
    




    @staticmethod
    def calculate_area(y_min, x_min, y_max, x_max):
        return (y_max - y_min) * (x_max - x_min)
    
    @staticmethod
    def fixaudict(box_dict):
        AU_labels = []
        AU_box_dict =defaultdict(list)
        aus= config.BP4D_USE_AU
        idx = 0
        for au in aus:
            for key,val in config.AU_BBOX.items():
                if int(au) in val:
                    for i in range(len(box_dict[key])):
                        AU_labels.append(au)
                        AU_box_dict[idx]=box_dict[key][i]
                        idx+=1
                    break
        return AU_labels,AU_box_dict

    @staticmethod
    def get_cropface_and_box(orig_img_path, mc_manager=None):
        #pdb.set_trace()
        key = orig_img_path
        orig_img = cv2.imread(config.DATA_ROOT_PATH+orig_img_path, cv2.IMREAD_COLOR) 
        if mc_manager is not None:
            try:

                result = mc_manager.get(key)

#                 if "crop_rect" in result:
                crop_rect = result["crop_rect"]
#                 else:
#                     landmark_dict = result["landmark_dict"]
#                     _, crop_rect = FaceMaskCropper.dlib_face_crop(orig_img, landmark_dict)
                AU_box_dict = result["AU_box_dict"]
                AU_labels = result["labels"]
                new_face = orig_img[crop_rect["top"]:crop_rect["top"] + crop_rect["height"],
                           crop_rect["left"]: crop_rect["left"] + crop_rect["width"], ...]
                new_face = cv2.resize(new_face, config.IMG_SIZE)

                target = {}
                target['bboxes'] = AU_box_dict
                target['labels_names'] = AU_labels
                #target['labels'] = list(map(lambda a: int(a[2:]),AU_labels))
                #target['labels'] = np.array(target['labels'])-1
                return np.array(new_face), target
            except Exception:
                pass

        landmark_dict, _ = FaceMaskCropper.landmark.landmark(orig_img, need_txt_img=False)
        if not isinstance(landmark_dict,dict):
            return 0, 0
        new_face, rect = FaceMaskCropper.dlib_face_crop(orig_img, landmark_dict)
        new_face = cv2.resize(new_face, config.IMG_SIZE)

        del orig_img
        AU_box_dict =defaultdict(list)
        for box_id in config.AU_BBOX_ROI.keys():
            mask = crop_face_mask_from_landmark(box_id, landmark_dict, new_face, rect, landmarker=FaceMaskCropper.landmark)
            connect_arr = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)  # mask shape = 1 x H x W
            component_num = connect_arr[0]
            label_matrix = connect_arr[1]
            # convert mask polygon to rectangle
            for component_label in range(1, component_num):

                row_col = list(zip(*np.where(label_matrix == component_label)))
                row_col = np.array(row_col)
                y_min_index = np.argmin(row_col[:, 0])
                y_min = row_col[y_min_index, 0]
                x_min_index = np.argmin(row_col[:, 1])
                x_min = row_col[x_min_index, 1]
                y_max_index = np.argmax(row_col[:, 0])
                y_max = row_col[y_max_index, 0]
                x_max_index = np.argmax(row_col[:, 1])
                x_max = row_col[x_max_index, 1]
                # same region may be shared by different AU, we must deal with it
                coordinates = [x_min, y_min, x_max, y_max]

                if y_min == y_max and x_min == x_max:  # 尖角处会产生孤立的单个点，会不会有一个mask
                    continue

                if FaceMaskCropper.calculate_area(y_min, x_min, y_max, x_max) / \
                        float(config.IMG_SIZE[0] * config.IMG_SIZE[1]) < 0.01:
                    continue
                AU_box_dict[box_id].append(coordinates)
                
            del label_matrix
            del mask
        AU_labels, AU_box_dict = FaceMaskCropper.fixaudict(AU_box_dict)
        
        if mc_manager is not None:
            try:
                save_dict = {"crop_rect":rect, "AU_box_dict":AU_box_dict, "landmark_dict":landmark_dict, "labels":AU_labels}
                mc_manager.set(key, save_dict)
            except Exception:
                pass
        target = {}
        target['bboxes'] = AU_box_dict
        target['labels_names'] = AU_labels
        #target['labels'] = list(map(lambda a: int(a[2:]),AU_labels))
        #target['labels'] = np.array(target['labels'])-1
        return np.array(new_face),target


def calculate_offset_polygon_arr(rect, new_face_image, polygon_arr):
    top, left, width, height = rect["top"], rect["left"], rect["width"], rect["height"]
    polygon_arr = polygon_arr.astype(np.float32)
    polygon_arr -= np.array([left, top])
    polygon_arr *= np.array([float(new_face_image.shape[1])/ width,
                             float(new_face_image.shape[0])/ height])
    polygon_arr = polygon_arr.astype(np.int32)
    return polygon_arr

def crop_face_mask_from_landmark(box_id, landmark, new_face_image, rect_dict, landmarker):
    mask = np.zeros(new_face_image.shape[:2], np.uint8)  # note that channel is LAST axis
    roi_polygons = landmarker.split_ROI(landmark)
    #pdb.set_trace()
    region_lst = config.AU_BBOX_ROI[box_id]
    #region_lst = config.AU_ROI[str(action_unit_no)]
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        if roi_no in region_lst:
            polygon_vertex_arr = calculate_offset_polygon_arr(rect_dict, new_face_image, polygon_vertex_arr)
            cv2.fillConvexPoly(mask, polygon_vertex_arr, 50)
    return mask