import os
import subprocess
import torch.distributed as dist



def init_distributed_mode():
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23700'
    dist.init_process_group('nccl', rank=0, world_size=2,init_method='env://') 
    
    
 