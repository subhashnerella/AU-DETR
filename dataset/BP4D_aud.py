import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from img_toolkit.face_bbox_cropper import FaceBboxCropper
from utils.box_utils import box_xyxy_to_cxcywh
from PIL import Image
import torchvision.transforms as T
import cv2
import config
import pdb
import json

#mean:tensor([0.4243, 0.2623, 0.2120]), std:tensor([0.1989, 0.1201, 0.0861])
d_transform = T.Compose([T.ToTensor(),T.Normalize([0.4243, 0.2623, 0.2120], [0.1989, 0.1201, 0.0861])])

class BP4D_dataset(Dataset):
    def __init__(self,df,split,data,transforms=d_transform):
        #pdb.set_trace()
        self.df = df
        self._transforms = transforms
        self.split_data(split)
        self.length = len(self.df)
        self.transform = transforms
        self.bboxer = FaceBboxCropper()
        self.data = data
        
    def split_data(self,split):
        assert len(split)>0
        self.df = self.df.loc[self.df.participant.isin(split)]
        self.df.reset_index(drop=True,inplace=True)
        #self.df = self.df.iloc[:10000]
        
        
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        #pdb.set_trace()
        img_path = self.df.iloc[idx][-2] 
        ids = np.array(self.df.iloc[idx,2:-2],np.bool)
        image = Image.open(config.DATA_ROOT_PATH+img_path)
        result = self.data[img_path]
        boxes = result['boxes']
        #labels = result['labels']
        target = self.processlabels(image,boxes,ids)
        target['image_id'] = torch.tensor([self.df.iloc[idx,0]])
        if self.transform:
            image = self.transform(image)  
        return image,target


    def processlabelss(self,image,boxes,labels,ids):
        #pdb.set_trace()
        #h,w,_ = image.shape
        w,h = image.size
        AUs_in_image = np.array(config.BP4D_USE_AU)[ids]
        boxes_AU = []; labels_au = []
        for label,value in zip(labels,boxes.values()):
            if str(label) in AUs_in_image:
                boxes_AU.append(value)
                labels_au.append(str(label))
        boxes_AU = torch.as_tensor(boxes_AU, dtype=torch.float32).reshape(-1, 4)
        boxes_AU = box_xyxy_to_cxcywh(boxes_AU)        
        boxes_AU = boxes_AU / torch.tensor([w, h, w, h], dtype=torch.float32)
        target = {}
        target['boxes'] = boxes_AU
        target['label_aus'] = torch.tensor(np.array(labels_au,dtype=np.int), dtype=torch.int64)
        labels = np.array(list(map(config.LABEL_MAP.get,labels_au)),np.int)
        target['labels'] =torch.tensor(labels, dtype=torch.int64) 
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        return target
    
    def processlabels(self,image,boxes,ids):
        #pdb.set_trace()
        w,h = image.size
        AUs_in_image = np.array(config.BP4D_USE_AU,dtype= np.int)[ids]
        labels_au = []; box_labels=[]; repeat = [];
        for aus,box in zip(config.AU_BOX_map.values(),boxes.values()):
            aus = list(set(aus) & set(AUs_in_image))
            n_aus = len(aus)
            if n_aus>0:
                labels_au.extend(aus)
                box_labels.append(box)
                repeat.append(n_aus)
        box_labels = torch.as_tensor(box_labels, dtype=torch.float32).reshape(-1, 4)
        box_labels = box_xyxy_to_cxcywh(box_labels)        
        box_labels = box_labels / torch.tensor([w, h, w, h], dtype=torch.float32)
        target = {}
        target['boxes'] = box_labels
        target['label_aus'] = torch.tensor(np.array(labels_au,dtype=np.int), dtype=torch.int64)
        labels = np.array(list(map(config.LABEL_MAP.get,labels_au)),np.int)
        target['labels'] =torch.tensor(labels, dtype=torch.int64)
        target['repeat'] = torch.tensor(repeat, dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        return target