import numpy as np
from glob import glob
import xarray as xr
import torch
from torch.utils.data import Dataset
import datetime

def get_analysis_date(file):
    return datetime.datetime.strptime(file.split('/')[-1], '%Y-%m-%dT%H')

class ECMWFDownscaleDataset(Dataset):

    def __init__(
        self,
        config
    ):
        
        self.files = [file for file in glob(f'{config.data.data_dir}/*') if '.idx' not in file]
        self.files.sort()
        self.downscale_factor = config.data.downscale_factor
        self.levels = config.data.input_levels
        self.lat_range = config.data.lat_range
        self.lon_range = config.data.lon_range
        self.surface_vars = config.data.input_surface_vars
        self.vertical_vars = config.data.input_vertical_vars
        self.static_surface_vars = config.data.input_static_surface_vars
        self.output_vars = config.data.output_vars
        self.n_timestamps = config.data.n_input_timestamps
    
    def __getitem__(self, index) -> dict[torch.Tensor]:

        # get the files to load
        file1 = self.files[index]
        if self.n_timestamps == 2:
            file2 = self.files[index + 1]
            d1 = get_analysis_date(file1)
            d2 = get_analysis_date(file2)
            assert d2 - d1 == datetime.timedelta(hours=6)

        # load the data
        ds1 = xr.open_dataset(file1, engine='cfgrib')
        if self.n_timestamps == 2:
            ds2 = xr.open_dataset(file2, engine='cfgrib')

        # select the lat  and lon range
        ds1 = ds1.sel(latitude=slice(self.lat_range[1], self.lat_range[0]), longitude=slice(self.lon_range[0], self.lon_range[1]))
        if self.n_timestamps == 2:
            ds2 = ds2.sel(latitude=slice(self.lat_range[1], self.lat_range[0]), longitude=slice(self.lon_range[0], self.lon_range[1]))


        # get the model level indices
        level_idxs = []
        for l in self.levels:
            level_idxs.append(np.argmin(np.abs(ds1['isobaricInhPa'].values - l)))

        # variables with vertical levels
        vert = {}
        dss = [ds1] if self.n_timestamps == 1 else [ds1, ds2]
        for i, ds in enumerate(dss):
            vert[i] = {}
            for var in self.vertical_vars:
                vert[i][var] = ds[var].values[level_idxs]
                if var == 'gh':
                    vert[i][var] *= 9.80665

        # surface only variables
        surf = {}
        for i, ds in enumerate(dss):
            surf[i] = {}
            for var in self.surface_vars:
                surf[i][var] = np.expand_dims(ds[var].values, axis=0)
                
        # static surface variables
        static = {}
        for var in self.static_surface_vars:
            static[var] = np.expand_dims(ds1[var].values, axis=0)

        # combine the data
        lat_idxs = np.arange(0, ds1.latitude.values.shape[0], self.downscale_factor)
        lon_idxs = np.arange(0, ds1.longitude.values.shape[0], self.downscale_factor)
        x_high_res_vals = []
        for i in vert.keys():
            for var in self.vertical_vars:
                x_high_res_vals.append(vert[i][var])
            for var in self.surface_vars:
                x_high_res_vals.append(surf[i][var])
        x_high_res = np.concatenate(x_high_res_vals, axis=0)
        x = x_high_res[:, lat_idxs, :][:, :, lon_idxs]
        y = np.concatenate([surf[0][var] for var in self.output_vars], axis=0)
        static_y = np.concatenate([static[var] for var in self.static_surface_vars], axis=0)
        static_x = static_y[:, lat_idxs, :][:, :, lon_idxs]

        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            'static_x': torch.from_numpy(static_x),
            'static_y': torch.from_numpy(static_y),
        }

    def __len__(self) -> int:
        return len(self.files) - 1
        
