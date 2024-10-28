import xarray as xr
from xarray_regrid import Grid, create_regridding_dataset
import numpy as np
from omegaconf import OmegaConf
import json
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(module)s %(levelname)s: %(message)s'
)

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
mu = np.array(mu).reshape(-1, 1, 1)
sigma = np.array(sigma).reshape(-1, 1, 1)

# get the indices of the output variables within the input tensor
idxs = []
offset = len(config.data.input_vertical_vars) * len(config.data.input_levels)
for var in config.data.output_vars:
    for i in range(len(config.data.input_surface_vars)):
        if config.data.input_surface_vars[i] == var:
            idxs.append(i + offset)
            break

# load the list of validation files to evaluate
with open('data/val_files.json', 'r') as f:
    val_files = json.load(f)

# step through the files and evaluate
vl = 0
for i, file in enumerate(val_files):
    

    # load the sample
    sample = np.load(file)

    # scale the input and output
    x = (sample['x'][idxs, :, :] - mu) / sigma
    y = (sample['y'] - mu) / sigma
    
    # get the lats and lons for the sample
    filename = file.split('/')[-1]
    chunks = filename.split('_')
    lat_s = int(chunks[1])
    lon_s = int(chunks[2])
    sz = chunks[3]
    sz = int(sz.split('.')[0])
    lat_e = lat_s + sz
    lon_e = lon_s + sz
    x_lats = np.arange(lat_e, lat_s, 0.2, dtype=np.float32)
    x_lons = np.arange(lon_s, lon_e, 0.2, dtype=np.float32)
    y_lats = np.arange(lat_e, lat_s, 0.1, dtype=np.float32)
    y_lons = np.arange(lon_s, lon_e, 0.1, dtype=np.float32)

    # generate an xarray dataset from x
    ds = xr.Dataset(
        {
            't2m': (['lat', 'lon'], x[0, :, :]),
            'u10': (['lat', 'lon'], x[1, :, :]),
            'v10': (['lat', 'lon'], x[2, :, :]),
            'msl': (['lat', 'lon'], x[3, :, :]),
        },
        coords={
            'lat': x_lats,
            'lon': x_lons,
        }
    )

    # regrid the dataset to the target grid
    d = y_lats[1] - y_lats[0]
    target_grid = Grid(
        north=y_lats.max(),
        east=y_lons.max(),
        south=x_lats.min(),
        west=y_lons.min(),
        resolution_lat=d,
        resolution_lon=d,
    )
    target_dataset = create_regridding_dataset(target_grid, lat_name='lat', lon_name='lon')
    new_ds = ds.regrid.linear(target_dataset)

    # calculate the loss
    pred = np.zeros_like(y)
    pred[0, :, :] = new_ds['t2m'].values
    pred[1, :, :] = new_ds['u10'].values
    pred[2, :, :] = new_ds['v10'].values
    pred[3, :, :] = new_ds['msl'].values
    l = np.mean((pred - y)**2)
    vl += l
    print(f'{i}/{len(val_files)}, loss: {l:.3f}', end='\r')

