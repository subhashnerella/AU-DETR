# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import pdb
import torch
import numpy as np
import config
import util.misc as utils
from collections import defaultdict
from sklearn.metrics import classification_report
import pandas as pd
import json
#from datasets.coco_eval import CocoEvaluator
#from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    #pdb.set_trace()
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #pdb.set_trace()
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
 
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        #pdb.set_trace()
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate_test(model, criterion, postprocessors, data_loader, device):
    #pdb.set_trace()
    model.eval()
    criterion.eval()
    all_gt=[];all_pred=[]
    for samples, targets in data_loader:
        samples = samples.to(device)
        results = model(samples)
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        results = postprocessors['bbox'](outputs,target_sizes)
        labels_pred = np.array([result['scores'].cpu().detach().numpy() for result in results],np.int)#B,Q,C
        labels_pred = np.bitwise_or.reduce(labels_pred[:,:,:-1], axis=1) #B,C
        all_pred.extend(labels_pred) 
        labels_gt= []
        for target in targets:
            label_inds = np.unique(target['labels'])
            temp = np.array([target['image_id']]+[0]*len(config.BP4D_USE_AU))
            temp[label_inds+1] =1
            labels_gt.append(temp)
        all_gt.extend(labels_gt)
    #pdb.set_trace()
    all_gt = np.vstack(utils.all_gather(all_gt))
    all_pred = np.vstack(utils.all_gather(all_pred))
    df = pd.DataFrame(columns=['image_id']+config.BP4D_USE_AU,data=all_gt)
    df.to_csv('output/trues_.csv',index=False) 
    df_pred = pd.DataFrame(columns = config.BP4D_USE_AU,data=all_pred)
    df_pred['image_id'] = df['image_id']
    df_pred.to_csv('output/preds_.csv',index=False) 
        
    all_gt = np.asarray(all_gt)[:,1:]
    all_pred = np.asarray(all_pred)
    all_gt = np.transpose(all_gt)
    all_pred = np.transpose(all_pred)
    report = defaultdict(dict)
    for true,pred,AU in zip(all_gt,all_pred,config.BP4D_USE_AU):
        reports = classification_report(true, pred,output_dict=True,labels=[0,1])
        try:
            precision = reports['1']['precision']
            accuracy = reports['accuracy']#['micro avg']['f1-score']
            recall = reports['1']['recall']
            specificity = reports['0']['recall']
            f1score = reports['1']['f1-score']
            support = reports['1']['support']

            report["f1"][AU] = f1score
            report["precision"][AU] = precision
            report["acc"][AU] = accuracy
            report["recall"][AU] = recall
            report["specificity"][AU] = specificity
            report['support'][AU] = support
        except:
            pass
    #pdb.set_trace()
    with open("output/BP4D_test_.json", "w") as file_obj:
        json.dump(report, file_obj)
        
        
        
        
        
        