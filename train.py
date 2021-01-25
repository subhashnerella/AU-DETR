import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
import time
import os
import util.misc as utils
import utils.misc as utils_s
import torch
from torch.utils.data import DataLoader, DistributedSampler
import dataset
from dataset.BP4D_aud import BP4D_dataset
from engine import evaluate, train_one_epoch,evaluate_test
from model import build_model
from utils.mc_manager import mcManager
import datetime
import pdb
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,help='gradient clipping max norm')


#backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

#transformer
 
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=7, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', default=False,
                    help="Disables auxiliary decoding losses (loss at each layer)")

# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")


# * Loss coefficients
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,help="Relative classification weight of the no-object class")

#dataset
parser.add_argument('--dataset_file', default='BP4D')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')

#multigpu
#parser.add_argument('--gpu', '-g', default="1,2")
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:9876', help='url used to set up distributed training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument("--local_rank", type=int)
parser.add_argument("--local_world_size", type=int, default=2)
args = parser.parse_args()

#python -m torch.distributed.launch --nproc_per_node=1 train.py
#pdb.set_trace()
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
utils.init_distributed_mode(args)    
# fix the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

model, criterion, postprocessors = build_model(args)
model.cuda(args.gpu)
 
model_without_ddp = model
#pdb.set_trace()
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]   

optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


df =  pd.read_csv('./dataset/BP4D_papercleaned.csv')
participants = np.unique(df.participant)
random.shuffle(participants)
split = {}
split['train'] = participants[:25]
split['val'] = participants[25:30]
split['test'] = participants[30:]
#memcached manager
mcmanager = mcManager()
if torch.distributed.get_rank() == 0:
     writer = SummaryWriter()
with open('preprocess1.json') as f:
    data = json.load(f)
dataset_train = BP4D_dataset(df,split['train'],data)
dataset_val = BP4D_dataset(df,split['val'],data)
dataset_test = BP4D_dataset(df,split['test'],data)

if args.distributed:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    sampler_test = DistributedSampler(dataset_test, shuffle=False)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                           collate_fn=utils.collate_fn, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                         drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                         drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

def plot_stats(stats,epoch,split='train'):
    for k,v in stats.items():
        writer.add_scalar(split+'_'+k, v, epoch + 1)

output_dir = Path(args.output_dir)
base_ds =None
print("Start training")
start_time = time.time()
for epoch in range(args.epochs):
    sampler_train.set_epoch(epoch)
    train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,args.clip_max_norm)
    lr_scheduler.step()
    if args.output_dir and  utils.is_main_process():
        plot_stats(train_stats,epoch)
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

    val_stats = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
    
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,'time':time.time()}
    if args.output_dir and utils.is_main_process():
        plot_stats(val_stats,epoch,'val')
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

with torch.no_grad():
    test_stats = evaluate_test(model,criterion,postprocessors,data_loader_test,device)
if torch.distributed.get_rank() == 0:
     writer.close()
    

