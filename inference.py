
import random
import numpy as np
import torch
import datetime
import xarray as xr
from xarray_regrid import Grid, create_regridding_dataset
from omegaconf import OmegaConf
from granitewxc.datasets.ecmwf import ECMWFDownscaleDataset
from granitewxc.utils.downscaling_model import get_finetune_model
import logging
import os
import subprocess
import matplotlib.pyplot as plt
import wget

from aws_utils import *

log = logging.getLogger(__name__)
if __name__ == '__main__':
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
config = OmegaConf.load('ecmwf_config.yaml')

# load the trained model
model = get_finetune_model(config, logger=None)
model.load_state_dict(torch.load('data/model.pt', map_location=torch.device('cpu')))
model.to(device)
model.eval()

def download_analysis(date):
    analysis_filename = date.strftime('D1D%m%d%H00%m%d%H001')
    if not os.path.isfile(f'data/{analysis_filename}'):
        try:
            download_file_s3('simulation.predictwind.com-ecmwf', f'{date.strftime("%Y")}/{date.strftime("%m")}/{analysis_filename}.bz2', f'data/{analysis_filename}.bz2', profile_name='pw')
            subprocess.run(['bzip2', '-d', f'data/{analysis_filename}.bz2'])
            if os.path.isfile(f'data/{analysis_filename}'):
                return f'data/{analysis_filename}'
        except:
            alt_date = date + datetime.timedelta(hours=8)
            if alt_date.month != date.month:
                try:
                    download_file_s3('simulation.predictwind.com-ecmwf', f'{alt_date.strftime("%Y")}/{alt_date.strftime("%m")}/{analysis_filename}.bz2', f'data/{analysis_filename}.bz2', profile_name='pw')
                    subprocess.run(['bzip2', '-d', f'data/{analysis_filename}.bz2'])
                    if os.path.isfile(f'data/{analysis_filename}'):
                        return f'data/{analysis_filename}'
                except:
                    log.error(f'Analysis file {analysis_filename} not found')
    else:
        return f'data/{analysis_filename}'
    
def download_aifs(issue_date, valid_date):
    hours = int((valid_date - issue_date).total_seconds() / 3600)
    ymd = issue_date.strftime('%Y%m%d')
    h = issue_date.strftime('%H')
    aifs_filename = f'{ymd}{h}0000-{hours}h-oper-fc.grib2'
    if not os.path.isfile(f'data/{aifs_filename}'):
        try:
            wget.download(f'https://data.ecmwf.int/forecasts/{ymd}/{h}z/aifs/0p25/oper/{aifs_filename}', out=f'data/{aifs_filename}')
            return f'data/{aifs_filename}'
        except:
            log.error(f'AIFS file {aifs_filename} not found')
    else:
        return f'data/{aifs_filename}'
    
def downscale_analysis(file1, file2, input_resolution, output_resolution, date):

    # load the file to downscale
    ds1 = xr.open_dataset(file1, engine='cfgrib')
    if file2 is not None:
        ds2 = xr.open_dataset(file2, engine='cfgrib')

    # select the vertical levels
    level_idxs = []
    for l in config.data.input_levels:
        level_idxs.append(np.argmin(np.abs(ds1['isobaricInhPa'].values - l)))
    ds1 = ds1.isel(isobaricInhPa=level_idxs)
    if file2 is not None:
        ds2 = ds2.isel(isobaricInhPa=level_idxs)

    # setup the downscaled dataset
    lat_vals = np.arange(-80, 80, output_resolution)
    lon_vals = np.arange(0, 360, output_resolution)
    lat_sz = lat_vals.shape[0]
    lon_sz = lon_vals.shape[0]
    downsale_ds = xr.Dataset(
        {
            't2m': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
            'u10': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
            'v10': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
            'msl': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
        },
        coords={
            'lat': lat_vals,
            'lon': lon_vals
        }
    )

    # step through the lat-lon windows
    for lat_s in range(-80, 80, config.data.window_sz):
        lat_e = lat_s + config.data.window_sz
        for lon_s in range(0, 360, config.data.window_sz):
            lon_e = lon_s + config.data.window_sz
            log.info(f'window: {lat_s} to {lat_e}, {lon_s} to {lon_e}')

            # select the window of interest
            output_lat_vals = np.arange(lat_s, lat_e, output_resolution)
            output_lon_vals = np.arange(lon_s, lon_e, output_resolution)
            lat_idxs = []
            lon_idxs = []
            for lat in output_lat_vals:
                lat_idxs.append(np.argmin(np.abs(ds1['latitude'].values - lat)))
            for lon in output_lon_vals:
                lon_idxs.append(np.argmin(np.abs(ds1['longitude'].values - lon)))
            w_ds1 = ds1.isel(latitude=lat_idxs, longitude=lon_idxs)
            if file2 is not None:
                w_ds2 = ds2.isel(latitude=lat_idxs, longitude=lon_idxs)

            # variables with vertical levels
            vert = {}
            dss = [w_ds1] if file2 is None else [w_ds1, w_ds2]
            for i, ds in enumerate(dss):
                vert[i] = {}
                for var in config.data.input_vertical_vars:
                    vert[i][var] = ds[var].values
                    if var == 'gh':
                        vert[i][var] *= 9.80665

            # surface only variables
            surf = {}
            for i, ds in enumerate(dss):
                surf[i] = {}
                for var in config.data.input_surface_vars:
                    surf[i][var] = np.expand_dims(ds[var].values, axis=0)
                    
            # static surface variables
            static = {}
            for var in config.data.input_static_surface_vars:
                static[var] = np.expand_dims(w_ds1[var].values, axis=0)

            # lat lon static vars
            n_lat = output_lat_vals.shape[0]
            n_lon = output_lon_vals.shape[0]
            lats = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            lons = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            for i in range(n_lat):
                lats[0, i, :] = output_lat_vals[i] / 90
            for i in range(n_lon):
                lons[0, :, i] = output_lon_vals[i] / 180 - 1
            static['lat'] = lats
            static['lon'] = lons

            # day of year static variables
            file = file1 if file2 is None else file2
            filename = file.split('/')[-1]
            
            y_day = date.timetuple().tm_yday
            static['cos_y_day'] = np.full((1, n_lat, n_lon), np.cos(2 * np.pi * y_day / 366), dtype=np.float32)
            static['sin_y_day'] = np.full((1, n_lat, n_lon), np.sin(2 * np.pi * y_day / 366), dtype=np.float32)

            # hour of day static variables
            h = date.hour / 24
            cos_h = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            sin_h = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            for i in range(n_lat):
                h_val = (h + output_lat_vals[i] / 360) % 1
                cos_h[0, i, :] = np.cos(2 * np.pi * h_val)
                sin_h[0, i, :] = np.sin(2 * np.pi * h_val)
            static['cos_hod'] = cos_h
            static['sin_hod'] = sin_h

            # combine the data
            lat_idxs = np.arange(0, output_lat_vals.shape[0], config.data.downscale_factor)
            lon_idxs = np.arange(0, output_lon_vals.shape[0], config.data.downscale_factor)
            x_high_res_vals = []
            for i in vert.keys():
                for var in config.data.input_vertical_vars:
                    x_high_res_vals.append(vert[i][var])
                for var in config.data.input_surface_vars:
                    x_high_res_vals.append(surf[i][var])
            x_high_res = np.concatenate(x_high_res_vals, axis=0)
            x = x_high_res[:, lat_idxs, :][:, :, lon_idxs]
            # y = np.concatenate([surf[0][var] for var in config.data.output_vars], axis=0)
            static_y = np.concatenate([static[var] for var in static.keys()], axis=0)
            static_x = static_y[:, lat_idxs, :][:, :, lon_idxs]

            # generate the model predictions
            x = np.expand_dims(x, axis=0)
            static_x = np.expand_dims(static_x, axis=0)
            static_y = np.expand_dims(static_y, axis=0)
            batch = {
                'x': torch.tensor(x).to(device),
                'static_x': torch.tensor(static_x).to(device),
                'static_y': torch.tensor(static_y).to(device),
            }
            with torch.no_grad():
                out = model(batch)
                log.info(f'out: {out.shape}')

            # add the predictions to the dataset
            lat_idx_s = np.argmin(np.abs(lat_vals - output_lat_vals[0]))
            lat_idx_e = lat_idx_s + output_lat_vals.shape[0]
            lon_idx_s = np.argmin(np.abs(lon_vals - output_lon_vals[0]))
            lon_idx_e = lon_idx_s + output_lon_vals.shape[0]
            for i, var in enumerate(config.data.output_vars):
                downsale_ds[var].values[lat_idx_s:lat_idx_e, lon_idx_s:lon_idx_e] = out[0, i].cpu().numpy()

    # save as a netcdf file
    downsale_ds.to_netcdf('data/downscaled.nc')
    return 'data/downscaled.nc'

def downscale_aifs(file, analysis_file, input_resolution, output_resolution, date):

    # load the file to downscale
    ds1 = xr.open_dataset(file, engine='cfgrib')

    # check if the longitude values are -180 to 180 or 0 to 360
    lon_0 = True
    if np.min(ds1['longitude'].values) < -179:
        lon_0 = False

    # load the analysis file and scale to the output resolution only including the required variables
    analysis = xr.open_dataset(analysis_file, engine='cfgrib')
    analysis_subset = xr.Dataset(
        {
            'z': (['latitude', 'longitude'], analysis['z'].values),
            'lsm': (['latitude', 'longitude'], analysis['lsm'].values),
        },
        coords={
            'latitude': analysis['latitude'].values,
            'longitude': analysis['longitude'].values
        }
    )
    new_grid = Grid(
        north=80,
        east=360,
        south=-80,
        west=0,
        resolution_lat=output_resolution,
        resolution_lon=output_resolution,
    )
    target_dataset = create_regridding_dataset(new_grid, lat_name='latitude', lon_name='longitude')
    analysis = analysis_subset.regrid.linear(target_dataset)

    # select the vertical levels
    level_idxs = []
    for l in config.data.input_levels:
        level_idxs.append(np.argmin(np.abs(ds1['isobaricInhPa'].values - l)))
    ds1 = ds1.isel(isobaricInhPa=level_idxs)

    # setup the downscaled dataset
    lat_vals = np.arange(-75, 75, output_resolution)
    lon_vals = np.arange(0, 350, output_resolution)
    lat_sz = lat_vals.shape[0]
    lon_sz = lon_vals.shape[0]
    downsale_ds = xr.Dataset(
        {
            't2m': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
            'u10': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
            'v10': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
            'msl': (['lat', 'lon'], np.zeros((lat_sz, lon_sz), dtype=np.float32)),
        },
        coords={
            'lat': lat_vals,
            'lon': lon_vals
        }
    )

    # step through the lat-lon windows
    for lat_s in range(-75, 75, 25):
        lat_e = lat_s + 25
        for lon_s in range(0, 350, 25):
            lon_e = lon_s + 25
            log.info(f'window: {lat_s} to {lat_e}, {lon_s} to {lon_e}')

            # select the window of interest
            input_lat_vals = np.arange(lat_s, lat_e, input_resolution)
            input_lon_vals = np.arange(lon_s, lon_e, input_resolution)
            lat_idxs = []
            lon_idxs = []
            for lat in input_lat_vals:
                lat_idxs.append(np.argmin(np.abs(ds1['latitude'].values - lat)))
            for lon in input_lon_vals:
                if not lon_0 and lon > 180:
                    lon -= 360
                lon_idxs.append(np.argmin(np.abs(ds1['longitude'].values - lon)))
            w_ds1 = ds1.isel(latitude=lat_idxs, longitude=lon_idxs)
            output_lat_vals = np.arange(lat_s, lat_e, output_resolution)
            output_lon_vals = np.arange(lon_s, lon_e, output_resolution)
            lat_idxs = []
            lon_idxs = []
            for lat in output_lat_vals:
                lat_idxs.append(np.argmin(np.abs(analysis['latitude'].values - lat)))
            for lon in output_lon_vals:
                lon_idxs.append(np.argmin(np.abs(analysis['longitude'].values - lon)))
            w_analysis = analysis.isel(latitude=lat_idxs, longitude=lon_idxs)

            # variables with vertical levels
            vert = {}
            dss = [w_ds1]
            for i, ds in enumerate(dss):
                vert[i] = {}
                for var in config.data.aifs_vertical_vars:
                    vert[i][var] = ds[var].values
                    if var == 'gh':
                        vert[i][var] *= 9.80665

            # surface only variables
            surf = {}
            for i, ds in enumerate(dss):
                surf[i] = {}
                for var in config.data.aifs_surface_vars:
                    surf[i][var] = np.expand_dims(ds[var].values, axis=0)
                    
            # static surface variables
            input_static = {}
            output_static = {}
            for var in config.data.input_static_surface_vars:
                # input_static[var] = np.expand_dims(w_ds1[var].values, axis=0)
                lat_idxs = np.arange(0, output_lat_vals.shape[0], config.data.downscale_factor, dtype=np.int32)
                lon_idxs = np.arange(0, output_lon_vals.shape[0], config.data.downscale_factor, dtype=np.int32)
                input_static[var] = np.expand_dims(w_analysis[var].values, axis=0)[:, lat_idxs, :][:, :, lon_idxs]
                output_static[var] = np.expand_dims(w_analysis[var].values, axis=0)

            # calculated static variables
            for static, s_lat_vals, s_lon_vals in [(input_static, input_lat_vals, input_lon_vals), (output_static, output_lat_vals, output_lon_vals)]:
                n_lat = s_lat_vals.shape[0]
                n_lon = s_lon_vals.shape[0]
                lats = np.zeros((1, n_lat, n_lon), dtype=np.float32)
                lons = np.zeros((1, n_lat, n_lon), dtype=np.float32)
                for i in range(n_lat):
                    lats[0, i, :] = s_lat_vals[i] / 90
                for i in range(n_lon):
                    lons[0, :, i] = s_lon_vals[i] / 180 - 1
                static['lat'] = lats
                static['lon'] = lons
                
                y_day = date.timetuple().tm_yday
                static['cos_y_day'] = np.full((1, n_lat, n_lon), np.cos(2 * np.pi * y_day / 366), dtype=np.float32)
                static['sin_y_day'] = np.full((1, n_lat, n_lon), np.sin(2 * np.pi * y_day / 366), dtype=np.float32)

                # hour of day static variables
                h = date.hour / 24
                cos_h = np.zeros((1, n_lat, n_lon), dtype=np.float32)
                sin_h = np.zeros((1, n_lat, n_lon), dtype=np.float32)
                for i in range(n_lat):
                    h_val = (h + s_lat_vals[i] / 360) % 1
                    cos_h[0, i, :] = np.cos(2 * np.pi * h_val)
                    sin_h[0, i, :] = np.sin(2 * np.pi * h_val)
                static['cos_hod'] = cos_h
                static['sin_hod'] = sin_h

            # combine the data
            x_vals = []
            for i in vert.keys():
                for var in config.data.aifs_vertical_vars:
                    x_vals.append(vert[i][var])
                for var in config.data.aifs_surface_vars:
                    x_vals.append(surf[i][var])
            x = np.concatenate(x_vals, axis=0)
            static_x = np.concatenate([input_static[var] for var in input_static.keys()], axis=0)
            static_y = np.concatenate([output_static[var] for var in output_static.keys()], axis=0)

            # generate the model predictions
            x = np.expand_dims(x, axis=0).astype(np.float32)
            static_x = np.expand_dims(static_x, axis=0).astype(np.float32)
            static_y = np.expand_dims(static_y, axis=0).astype(np.float32)
            batch = {
                'x': torch.tensor(x).to(device),
                'static_x': torch.tensor(static_x).to(device),
                'static_y': torch.tensor(static_y).to(device),
            }
            with torch.no_grad():
                out = model(batch)
                log.info(f'out: {out.shape}')

            # add the predictions to the dataset
            lat_idx_s = np.argmin(np.abs(lat_vals - output_lat_vals[0]))
            lat_idx_e = lat_idx_s + output_lat_vals.shape[0]
            lon_idx_s = np.argmin(np.abs(lon_vals - output_lon_vals[0]))
            lon_idx_e = lon_idx_s + output_lon_vals.shape[0]
            log.info(f'lat_idx_s: {lat_idx_s}, lat_idx_e: {lat_idx_e}, lon_idx_s: {lon_idx_s}, lon_idx_e: {lon_idx_e}')
            for i, var in enumerate(config.data.output_vars):
                downsale_ds[var].values[lat_idx_s:lat_idx_e, lon_idx_s:lon_idx_e] = out[0, i].cpu().numpy()

    # save as a netcdf file
    downsale_ds.to_netcdf('data/downscaled.nc')
    return 'data/downscaled.nc'

def plot(file):
    ds = xr.open_dataset(file)
    for var in config.data.output_vars:
        plt.figure()
        ds[var].plot()
        plt.title(var)

def gen_interpolated_analysis(file):

    ds = xr.open_dataset(file, engine='cfgrib')

    # select the low resolution grid
    lat_vals = np.arange(-80, 80, 0.2)
    lon_vals = np.arange(0, 360, 0.2)
    lat_idxs = []
    lon_idxs = []
    for lat in lat_vals:
        lat_idxs.append(np.argmin(np.abs(ds['latitude'].values - lat)))
    for lon in lon_vals:
        lon_idxs.append(np.argmin(np.abs(ds['longitude'].values - lon)))
    
    # create a new dataset with the low resolution grid
    low_res_ds = xr.Dataset(
        {
            't2m': (['latitude', 'longitude'], ds['t2m'].values[lat_idxs, :][:, lon_idxs]),
            'u10': (['latitude', 'longitude'], ds['u10'].values[lat_idxs, :][:, lon_idxs]),
            'v10': (['latitude', 'longitude'], ds['v10'].values[lat_idxs, :][:, lon_idxs]),
            'msl': (['latitude', 'longitude'], ds['msl'].values[lat_idxs, :][:, lon_idxs]),
        },
        coords={
            'latitude': lat_vals,
            'longitude': lon_vals
        }
    )

    # create the target grid
    new_grid = Grid(
        north=80,
        east=360,
        south=-80,
        west=0,
        resolution_lat=0.1,
        resolution_lon=0.1,
    )
    target_dataset = create_regridding_dataset(new_grid, lat_name='latitude', lon_name='longitude')

    # interpolate the dataset
    downscale_ds = low_res_ds.regrid.linear(target_dataset)
    return downscale_ds

def plot_interpolated_analysis(file):
    
    # generate the interpolated dataset
    downscale_ds = gen_interpolated_analysis(file)

    # plot the interpolated dataset
    for var in config.data.output_vars:
        plt.figure()
        downscale_ds[var].plot()
        plt.title(var)

def compare_rmse(ds_file, analysis_file):

    # load the datasets
    downscale_ds = xr.open_dataset(ds_file)
    interp_ds = gen_interpolated_analysis(analysis_file)
    analysis = xr.open_dataset(analysis_file, engine='cfgrib')
    lat_vals = np.arange(-80, 80, 0.1)
    lon_vals = np.arange(0, 360, 0.1)
    lat_idxs = []
    lon_idxs = []
    for lat in lat_vals:
        lat_idxs.append(np.argmin(np.abs(analysis['latitude'].values - lat)))
    for lon in lon_vals:
        lon_idxs.append(np.argmin(np.abs(analysis['longitude'].values - lon)))
    analysis = analysis.isel(latitude=lat_idxs, longitude=lon_idxs)

    # calculate the RMSE for each variable
    for var in config.data.output_vars:
        rmse_interp = np.sqrt(np.nanmean((interp_ds[var].values - analysis[var].values) ** 2))
        rmse_downscale = np.sqrt(np.nanmean((downscale_ds[var].values - analysis[var].values) ** 2))
        log.info(f'RMSE for {var}, interpolated: {rmse_interp}, downscale: {rmse_downscale}')

if __name__ == '__main__':
    ic_date = datetime.datetime(2024, 11, 15, tzinfo=datetime.timezone.utc)
    date = datetime.datetime(2024, 11, 15, 6, tzinfo=datetime.timezone.utc)
    analysis_file = download_analysis(date)
    aifs_file = download_aifs(ic_date, date)
    input_resolution = 0.25
    output_resolution = 0.125
    # downscale_analysis(analysis_file, None, input_resolution, output_resolution, date)
    downscale_aifs(aifs_file, analysis_file, input_resolution, output_resolution, date)
    plot('data/downscaled.nc')
    plot_interpolated_analysis(analysis_file)
    plt.show()
    # compare_rmse('data/downscaled.nc', analysis_file)