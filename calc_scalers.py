from glob import glob
import xarray as xr
import os
import json

# set variables to get stats for
surface_vars = ['t2m', 'msl', 'u10', 'v10']
vertical_vars = ['t', 'u', 'v', 'w', 'q', 'gh']
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
static_surface_vars = ['lsm', 'z']

# initialize data dict
if os.path.exists('scalers.json'):
    with open('scalers.json', 'r') as f:
        data = json.load(f)

        # sum the values so more can be added and they can be averaged at the end
        for k, v in data.items():
            if k == 'files':
                continue
            v['mu'] *= v['n']
            v['sigma'] *= v['n']
else:
    data = {'files': []}

# get stats for each file, adding to the data dict
files = glob('/Volumes/ChrisHD/granite-wxc/ecmwf/*')
files = [file for file in glob('/Volumes/ChrisHD/granite-wxc/ecmwf/*') if '.idx' not in file]
for i, file in enumerate(files):
    print(f'{i+1}/{len(files)}')

    filename = file.split('/')[-1]
    if filename in data['files']:
        continue
    else:
        data['files'].append(filename)
    ds = xr.open_dataset(file, engine='cfgrib')
    for var in surface_vars:
        if var not in data:
            data[var] = {'mu': 0, 'sigma': 0, 'n': 0}
        data[var]['mu'] += float(ds[var].values.mean())
        data[var]['sigma'] += float(ds[var].values.std())
        data[var]['n'] += 1
    for var in vertical_vars:
        for level in levels:
            name = f'{var}_{level}'
            if name not in data:
                data[name] = {'mu': 0, 'sigma': 0, 'n': 0}
            data[name]['mu'] += float(ds[var].sel(isobaricInhPa=level).values.mean())
            data[name]['sigma'] += float(ds[var].sel(isobaricInhPa=level).values.std())
            data[name]['n'] += 1
    for var in static_surface_vars:
        if var not in data:
            data[var] = {}
            data[var]['mu'] = float(ds[var].values.mean())
            data[var]['sigma'] = float(ds[var].values.std())
            data[var]['n'] = 1

# average the values
for k, v in data.items():
    if k == 'files':
        continue
    v['mu'] /= v['n']
    v['sigma'] /= v['n']

# save the data dict
with open('scalers.json', 'w') as f:
    json.dump(data, f, indent=4)