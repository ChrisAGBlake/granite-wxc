
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from granitewxc.datasets.ecmwf import ECMWFDownscaleDataset
from granitewxc.utils.downscaling_model import get_finetune_model

torch.jit.enable_onednn_fusion(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config = OmegaConf.load('ecmwf_config.yaml')
dataset = ECMWFDownscaleDataset(config)
dataloader = DataLoader(dataset, batch_size=1)
model = get_finetune_model(config, logger=None)
batch = next(iter(dataloader))
for k, v in batch.items():
    print(k, v.shape)
with torch.no_grad():
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch)
print(out.shape)