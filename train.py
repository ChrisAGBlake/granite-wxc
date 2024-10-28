
import random
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from granitewxc.datasets.ecmwf import ECMWFDownscaleDataset
from granitewxc.utils.downscaling_model import get_finetune_model
import wandb
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(module)s %(levelname)s: %(message)s'
)

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

# load the config
config = OmegaConf.load('ecmwf_config.yaml')

# initialize wandb
wandb.login()
wandb.init(entity='chris_blake', project='pw-downscale', config=config)

# setup the datasets and dataloaders
files = list(glob(f'{config.data.parsed_data_dir}/*'))
random.shuffle(files)
split = int(len(files) * config.train.train_split)
train_files = files[:split]
val_files = files[split:]
train_dataset = ECMWFDownscaleDataset(train_files)
val_dataset = ECMWFDownscaleDataset(val_files)
train_dataloader = DataLoader(train_dataset, batch_size=config.train.bach_size, shuffle=True, num_workers=config.train.dl_num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=config.train.bach_size, shuffle=False, num_workers=config.train.dl_num_workers)

# setup the model
model = get_finetune_model(config, logger=None)
model.to(device)

# setup the loss function and optimizer
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

for i in range(config.train.num_epochs):

    # train
    tl = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch)
        l = loss(out, batch['y'])
        l.backward()
        optimizer.step()
        tl += l.item()
    log.info(f'epoch: {i}, train loss: {tl}')
    
    # evaluate
    vl = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            l = loss(out, batch['y'])
            vl += l.item()
    log.info(f'epoch: {i}, valloss: {vl}')
    model.train()

    # log to wandb
    wandb.log({'train_loss': tl, 'val_loss': vl})