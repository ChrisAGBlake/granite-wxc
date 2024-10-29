
import random
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from granitewxc.datasets.ecmwf import ECMWFDownscaleDataset
from granitewxc.utils.downscaling_model import get_finetune_model
import wandb
import json
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

# setup the scalers
with open('scalers.json', 'r') as f:
    scalers = json.load(f)
mu = []
sigma = []
for var in config.data.output_vars:
    mu.append(scalers[var]['mu'])
    sigma.append(scalers[var]['sigma'])
mu = torch.tensor(mu).reshape(1, -1, 1, 1).to(device)
sigma = torch.tensor(sigma).reshape(1, -1, 1, 1).to(device)

# initialize wandb
wandb.login()
wandb.init(entity='chris_blake', project='pw-downscale', config=config)

# setup the datasets and dataloaders
files = list(glob(f'{config.data.parsed_data_dir}/*'))
random.shuffle(files)
split = int(len(files) * config.train.train_split)
train_files = files[:split]
val_files = files[split:]
with open('data/train_files.json', 'w') as f:
    json.dump(train_files, f)
with open('data/val_files.json', 'w') as f:
    json.dump(val_files, f)
train_dataset = ECMWFDownscaleDataset(train_files)
val_dataset = ECMWFDownscaleDataset(val_files)
train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.dl_num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.dl_num_workers)

# setup the model
model = get_finetune_model(config, logger=None)
model.to(device)
    
# setup the loss function and optimizer
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
print('train dataset size:', len(train_dataset))
print('val dataset size:', len(val_dataset))

for n in range(config.train.num_epochs):

    # train
    tl = 0
    for i, batch in enumerate(train_dataloader):

        # get the predictions
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        # scale the predictions and targets
        out = (out - mu) / sigma
        y = (batch['y'] - mu) / sigma

        # get the loss
        l = loss(out, y)
        l.backward()

        # update the weights
        if (i + 1) % config.train.grad_accum_steps == 0 or i + 1 == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()

        # logging
        tl += l.item()
        print(i, l.item(), end='\r')
        if (i + 1) % 100 == 0:
            wandb.log({'train_loss': tl / 100})
            tl = 0
    
    # evaluate
    vl = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:

            # get the predictions
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)

            # scale the predictions and targets
            out = (out - mu) / sigma
            y = (batch['y'] - mu) / sigma

            # get the loss
            l = loss(out, y)
            vl += l.item()
    vl /= len(val_dataloader)
    wandb.log({'val_loss': vl})
    log.info(f'epoch: {n}, val loss: {vl}')

    # save the model and optimiser state
    torch.save(model.state_dict(), f'data/model_{n}.pt')
    torch.save(optimizer.state_dict(), f'data/optimizer_{n}.pt')
    model.train()
