# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
from util.misc import all_gather
from util.box_ops import box_iou
import config

class Evaluator():
    def __init__(self):
        self.params = Params(iouType=iouType)
        self.img_ids = []
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.maxdets = 
        
    def update(self,gts,preds):
        img_ids = list(np.unique(list(gts.keys())))
        
        self.img_ids.extend(img_ids)
        img_ids, eval_imgs = evaluate()
        
        
        for img_id,value in preds.items():
            scores = value['scores'];labels = value['labels']
            boxes = value['boxes']
            aus = config.BP4D_USE_AU[labels] 
            [self._dts[img_id,au].append({'box':box, 'score':score, 'id':i+1}) for i,(au,box,score) in enumerate(zip(aus,boxes,scores))]
        
        for gt in gts:
            img_id = gt['image_id']; labels= gt['labels']
            boxes = value['boxes']
            aus = config.BP4D_USE_AU[labels] 
            [self._gts[img_id,au].append({'box':box},'id':i+1) for i,(au,box,score) in enumerate(zip(aus,boxes,scores))]
            
    def computeIoU(self, imgId, au):
        gt = self._gts[imgId,au]
        dt = self._dts[imgId,au]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
        ious,_ = box_iou(d,g)
        return ious  
    
    def evaluate(self):
        """
        Run per image
        """
        tic = time.time()
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        catIds = p.catIds if p.useCats else [-1]
        self.ious = {(imgId, au): computeIoU(imgId, au) \
                        for imgId in p.imgIds
                        for au in aus}
        self.evalImgs = [evaluateImg(imgId, au, maxDet) for au in aus for imgId in p.imgIds]
        toc = time.time()
       
    def evaluate_img(self,imgid,au,maxdets)
        gts = self._gts[imgId,au]
        dts = self._dts[imgId,au]
        if len(gts) == 0 or len(dts) == 0:
            return []
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')        
        dt = [dt[i] for i in dtind[0:maxDet]]
        ious = self.ious[imgId, au]
        G = len(gt)
        D = len(dt) 
        #maxDet = 
        gtd = np.zeros(G,D)
        
        
        
        
        
    def evaluate_img(self,imgid,au,maxdets):
        p = self.params
        gts = self._gts[imgId,au]
        dts = self._dts[imgId,au]
        if len(gts) == 0 or len(dts) == 0:
            return []
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        ious = self.ious[imgId, au]
        
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)      
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                iou = min([t,1-1e-10])
                m=-1
                for gind, g in enumerate(gt):     
                    if gtm[tind,gind]>0 :
                        continue                    
                    if ious[dind,gind] < iou:
                        continue
                    iou=ious[dind,gind]
                    m=gind    
                if m ==-1:
                    continue
                dtm[tind,dind]  = gt[m]['id']
                gtm[tind,m]     = d['id']  
                
        return {
                'image_id':     imgId,
                'au':           au,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
               }
    

        
    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs




    

    

            
            
        
        
        
class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1


    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType

        

        
#if __name__=='__main__':
    