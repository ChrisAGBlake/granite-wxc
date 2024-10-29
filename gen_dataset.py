import numpy as np
from glob import glob
import xarray as xr
import datetime
from omegaconf import OmegaConf
import json
import os
import logging

from aws_utils import *

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(module)s %(levelname)s: %(message)s'
)

def get_analysis_date(file):
    return datetime.datetime.strptime(file.split('/')[-1], '%Y-%m-%dT%H')

def parse_file(config, file1, file2, upload):

    # initialize the s3 client
    if upload:
        s3 = connect_s3(config.aws.profile_name)

    # load the data
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

    # step through the lat-lon windows
    for lat_s in range(-80, 80, config.data.window_sz):
        lat_e = lat_s + config.data.window_sz
        for lon_s in range(0, 360, config.data.window_sz):
            lon_e = lon_s + config.data.window_sz
            log.info(f'window: {lat_s} to {lat_e}, {lon_s} to {lon_e}')

            # select the window of interest
            lat_vals = np.arange(lat_s, lat_e, 0.1)
            lon_vals = np.arange(lon_s, lon_e, 0.1)
            lat_idxs = []
            lon_idxs = []
            for lat in lat_vals:
                lat_idxs.append(np.argmin(np.abs(ds1['latitude'].values - lat)))
            for lon in lon_vals:
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
            n_lat = lat_vals.shape[0]
            n_lon = lon_vals.shape[0]
            lats = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            lons = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            for i in range(n_lat):
                lats[0, i, :] = lat_vals[i] / 90
            for i in range(n_lon):
                lons[0, :, i] = lon_vals[i] / 180 - 1
            static['lat'] = lats
            static['lon'] = lons

            # day of year static variables
            file = file1 if file2 is None else file2
            filename = file.split('/')[-1]
            d = datetime.datetime.strptime(filename, '%Y-%m-%dT%H').replace(tzinfo=datetime.timezone.utc)
            y_day = d.timetuple().tm_yday
            static['cos_y_day'] = np.full((1, n_lat, n_lon), np.cos(2 * np.pi * y_day / 366), dtype=np.float32)
            static['sin_y_day'] = np.full((1, n_lat, n_lon), np.sin(2 * np.pi * y_day / 366), dtype=np.float32)

            # hour of day static variables
            h = d.hour / 24
            cos_h = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            sin_h = np.zeros((1, n_lat, n_lon), dtype=np.float32)
            for i in range(n_lat):
                h_val = (h + lat_vals[i] / 360) % 1
                cos_h[0, i, :] = np.cos(2 * np.pi * h_val)
                sin_h[0, i, :] = np.sin(2 * np.pi * h_val)
            static['cos_hod'] = cos_h
            static['sin_hod'] = sin_h

            # combine the data
            lat_idxs = np.arange(0, lat_vals.shape[0], config.data.downscale_factor)
            lon_idxs = np.arange(0, lon_vals.shape[0], config.data.downscale_factor)
            x_high_res_vals = []
            for i in vert.keys():
                for var in config.data.input_vertical_vars:
                    x_high_res_vals.append(vert[i][var])
                for var in config.data.input_surface_vars:
                    x_high_res_vals.append(surf[i][var])
            x_high_res = np.concatenate(x_high_res_vals, axis=0)
            x = x_high_res[:, lat_idxs, :][:, :, lon_idxs]
            y = np.concatenate([surf[0][var] for var in config.data.output_vars], axis=0)
            static_y = np.concatenate([static[var] for var in static.keys()], axis=0)
            static_x = static_y[:, lat_idxs, :][:, :, lon_idxs]

            # save the files
            filename = f'{file1.split("/")[-1]}_{lat_s}_{lon_s}_{config.data.window_sz}.npz'
            file = f'{config.data.parsed_data_dir}/{filename}'
            np.savez_compressed(
                file,
                x=x,
                y=y,
                static_x=static_x,
                static_y=static_y,
            )

            # upload
            if upload:
                upload_file_s3(file, config.aws.s3_bucket, f'{config.aws.parsed_data_dir}/{filename}', s3_client=s3)

def main(upload):

    # load config
    config = OmegaConf.load('ecmwf_config.yaml')

    # create any required directories
    for d in ['data', config.data.parsed_data_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # initialise list of already parsed files
    if upload:
        try: 
            download_file_s3(config.aws.bucket, 'parsed_files.json', 'data/parsed_files.json', profile_name=config.aws.profile_name)
        except:
            pass
    if os.path.exists('data/parsed_files.json'):
        with open('data/parsed_files.json', 'r') as f:
            parsed_files = json.load(f)
    else:
        parsed_files = []

    # get the list of files to parse
    files = [file for file in glob(f'{config.data.data_dir}/*') if '.idx' not in file]
    files.sort()

    # parse the files
    for i in range(len(files)):
        file1 = files[i]
        if i < len(files) - 1 and config.data.n_input_timestamps == 2:
            file2 = files[i+1]
            d1 = get_analysis_date(file1)
            d2 = get_analysis_date(file2)
            assert d2 - d1 == datetime.timedelta(hours=6)
        else:
            file2 = None
        
        # check if the file has already been parsed
        filename = file1.split('/')[-1]
        if filename in parsed_files:
            continue

        # parse the file(s)
        log.info(f'parsing: {files[i]}, {i+1} of {len(files)}')
        parse_file(config, file1, file2, upload)

        # add the file to the list of parsed files
        parsed_files.append(filename)
        with open('data/parsed_files.json', 'w') as f:
            json.dump(parsed_files, f)
        if upload:
            upload_file_s3('data/parsed_files.json', config.aws.bucket, 'parsed_files.json', profile_name=config.aws.profile_name)

if __name__ == '__main__':
    main(False)