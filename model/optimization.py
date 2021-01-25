import torch
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from pulp import *
from itertools import product
import config
import numpy as np
import pdb
class MultilabelMatcher(nn.Module):
 
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    def forward(self,outputs,targets):
        #pdb.set_trace()
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() #cost to optimize
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_aus = torch.cat([v["label_aus"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        repeat = torch.cat([v["repeat"] for v in targets])
        
        
        # augment tgt_bbox to match au classes
        aug_tgt_bbox = torch.repeat_interleave(tgt_bbox,repeat,dim=0)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, aug_tgt_bbox, p=1)
        
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(aug_tgt_bbox))        
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]
        
        indices = [optimize(c[i],get_capacity(targets.tolist())) for i, (c,targets) in enumerate(zip(C.split(sizes, -1),tgt_aus.cpu().split(sizes)))]
        #pdb.set_trace()
        return indices
        #return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
def optimize(cost,targets):
    groups,capacity = get_constraintdata(targets)
    #pdb.set_trace()
    queries,targets = cost.shape
    objective = LpProblem("Optimize_AU_allocation",LpMinimize)
    x = LpVariable.dicts("assignment", product(range(queries), range(targets)), 0, 1, 'Integer')
    y = LpVariable.dicts("assignment", product(range(queries), range(len(groups))), 0, 1, 'Integer')
    
    objective += lpSum(cost[i, j] * x[i, j] for i in range(queries) for j in range(targets))
    
    for j in range(targets):
        condition1 = lpSum( x[i, j] for i in range(queries)) == 1
        objective+= condition1
    for i in range(queries):
        condition2 = lpSum(x[i, j] for j in range(targets)) <= capacity
        objective+= condition2  
        
    for i in range(queries):
        for k in range(len(groups)):
            objective += lpSum(x[i,j] for j in groups[k]) <= y[i,k]*capacities[k]
    
    for k in range(len(groups)):
        objective += lpSum(y[i,k] for i in range(queries)) == 1
        
    
    objective.solve(PULP_CBC_CMD(msg=0))
    
    x = torch.as_tensor([ value(x[i, j]) for i in range(queries) for j in range(targets)])
    x = x.view((queries,targets))
    return torch.nonzero(x,as_tuple=True)
   
    
def get_constraintdata(targets):
    #pdb.set_trace()
    capacities = []
    groups = []
    max_capacity = 0
    for value in list(config.AU_BOX_map.values()):
        group = list(set(targets) & set(value))
        groups.append(group)
        length = len(group)
        capacities.extend(length)
        if length > max_capacity:        
            max_capacity = length
    return max_capacity,capacities,groups
                   
                   
def build_matcher(args):
    return MultilabelMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)                   
                   
        
        
        
        
        
        
        
        
        
        
        
