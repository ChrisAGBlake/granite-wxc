import xarray as xr
from xarray_regrid import Grid, create_regridding_dataset
import numpy as np
from omegaconf import OmegaConf
import json
from glob import glob
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

# step through the files and evaluate
vars = config.data.output_vars
files = [file for file in glob(f'{config.data.data_dir}/*') if '.idx' not in file]
for file in files:

    # load the grib file
    ds = xr.open_dataset(file, engine='cfgrib')

    # select the variables of interest
    ds = ds[vars]

    # select the latitudes and longitudes of interest
    x_lats = np.arange(-80, 80.05, 0.2, dtype=np.float32)
    x_lons = np.arange(0, 360, 0.2, dtype=np.float32)
    y_lats = np.arange(-80, 80.05, 0.1, dtype=np.float32)
    y_lons = np.arange(0, 360, 0.1, dtype=np.float32)
    y_lats_idxs = []
    for lat in y_lats:
        y_lats_idxs.append(np.argmin(np.abs(ds['latitude'].values - lat)))
    y_lons_idxs = []
    for lon in y_lons:
        y_lons_idxs.append(np.argmin(np.abs(ds['longitude'].values - lon)))
    ds = ds.isel(latitude=y_lats_idxs, longitude=y_lons_idxs)

    # scale
    for var in vars:
        ds[var].values = (ds[var].values - scalers[var]['mu']) / scalers[var]['sigma']

    # generate target array
    target = np.zeros((len(vars), len(y_lats), len(y_lons)), dtype=np.float32)
    for i, var in enumerate(vars):
        target[i, :, :] = ds[var].values.copy()

    # generate the low resolution input
    x_lats_idxs = []
    for lat in x_lats:
        x_lats_idxs.append(np.argmin(np.abs(ds['latitude'].values - lat)))
    x_lons_idxs = []
    for lon in x_lons:
        x_lons_idxs.append(np.argmin(np.abs(ds['longitude'].values - lon)))
    inp_ds = ds.isel(latitude=x_lats_idxs, longitude=x_lons_idxs)

    # regrid the dataset to the target grid
    target_grid = Grid(
        north=80.05,
        east=360,
        south=-80,
        west=0,
        resolution_lat=0.1,
        resolution_lon=0.1,
    )
    target_dataset = create_regridding_dataset(target_grid, lat_name='latitude', lon_name='longitude')
    new_ds = inp_ds.regrid.linear(target_dataset)

    # check the grids match in size
    if new_ds['latitude'].values.shape[0] != y_lats.shape[0]:
        print('latitudes do not match')
        print(new_ds['latitude'].values.shape[0], y_lats.shape[0])
        continue
    if new_ds['longitude'].values.shape[0] != y_lons.shape[0]:
        print('longitudes do not match')
        print(new_ds['longitude'].values.shape[0], y_lons.shape[0])
        continue

    # generate the pred array
    pred = np.zeros((len(vars), len(y_lats), len(y_lons)), dtype=np.float32)
    for i, var in enumerate(vars):
        pred[i, :, :] = new_ds[var].values.copy()

    # calculate the loss
    l = np.mean((pred - target)**2)
    print(f'{file}, loss: {l:.3f}')

