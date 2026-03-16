# Dataset

The training dataset is **not included in this repository** (195 GB, 7313 files).

---

## Origin

This dataset was prepared by **PI Pedram Hassandazeh's group (Geophysical Sciences)**
as part of the Pangu S2S weather forecasting project. It is a pre-processed extract
of ERA5 reanalysis data covering **1979–2018**, reformatted into the Pangu S2S HDF5
schema (6-hourly snapshots, 1°×1° global grid, 180×360).

**To request access**, contact the dataset maintainer directly.

---

## Option 1 — Get the Pre-processed Dataset (Recommended)

If you have access to the group's storage, the dataset is located at:

```
/home/rdesouz4/scratchrdesouz4/b300/pangus2s/dataset/
```

Copy or symlink it to the `dataset/` directory of this repo:

```bash
# Symlink (no copy, saves space)
ln -s /home/rdesouz4/scratchrdesouz4/b300/pangus2s/dataset/* \
      /path/to/b300-benchmarks/dataset/

# Or copy just the years needed for a short benchmark run (1979–1980 = ~5 GB)
rsync -av --include='1979_*.h5' --include='1980_*.h5' \
          --include='*.nc' --exclude='*' \
    /home/rdesouz4/scratchrdesouz4/b300/pangus2s/dataset/ \
    /path/to/b300-benchmarks/dataset/
```

Then update `training/config/exp1_dsai.yaml` to point to your dataset path:
```yaml
data_dir: '/path/to/b300-benchmarks/dataset'
train_year_start: 1979
train_year_end: 1980   # small range for a benchmark run
```

---

## Option 2 — Download ERA5 and Generate from Scratch

If you do not have access to the pre-processed files, you can generate an
equivalent dataset from raw ERA5 using the **Copernicus Climate Data Store (CDS)**.

### Step 1 — Install the CDS API client

```bash
pip install cdsapi netcdf4 h5py xarray
```

Register at https://cds.climate.copernicus.eu and place your API key in `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API-KEY>
```

### Step 2 — Download ERA5 pressure-level fields

```python
import cdsapi

c = cdsapi.Client()

# Upper-air variables (17 pressure levels, 6-hourly)
c.retrieve('reanalysis-era5-pressure-levels', {
    'product_type': 'reanalysis',
    'variable': [
        'temperature', 'u_component_of_wind', 'v_component_of_wind',
        'specific_humidity', 'geopotential',
    ],
    'pressure_level': [
        '5','10','20','30','50','70','100','150',
        '250','300','400','500','600','700','850','925','1000',
    ],
    'year':  [str(y) for y in range(1979, 2019)],
    'month': [f'{m:02d}' for m in range(1, 13)],
    'day':   [f'{d:02d}' for d in range(1, 32)],
    'time':  ['00:00', '06:00', '12:00', '18:00'],
    'grid':  ['1.0', '1.0'],  # 1°×1° global
    'format': 'netcdf',
}, 'era5_upper_air_1979-2018.nc')
```

```python
# Surface variables (6-hourly)
c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'variable': [
        '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
        'mean_sea_level_pressure', 'surface_pressure',
        'total_precipitation', 'mean_top_net_long_wave_radiation_flux',
        'volumetric_soil_water_layer_1', 'soil_temperature_level_1',
        'skin_temperature', 'sea_surface_temperature',
        'land_sea_mask', 'geopotential',
        'toa_incident_solar_radiation',
    ],
    'year':  [str(y) for y in range(1979, 2019)],
    'month': [f'{m:02d}' for m in range(1, 13)],
    'day':   [f'{d:02d}' for d in range(1, 32)],
    'time':  ['00:00', '06:00', '12:00', '18:00'],
    'grid':  ['1.0', '1.0'],
    'format': 'netcdf',
}, 'era5_surface_1979-2018.nc')
```

### Step 3 — Convert to Pangu S2S HDF5 format

The model expects one HDF5 file per 6-hourly timestep, named `YYYY_NNNN.h5`
where `NNNN` is the zero-padded time index within the year (0000–1459 for a
non-leap year, 0000–1463 for a leap year).

Each file has keys matching the variable groups in `exp1_dsai.yaml`. Refer to
`training/utils/data_loader_multifiles.py` for the exact read logic and expected
array shapes.

### Step 4 — Compute normalization statistics

The `.nc` statistics files (mean, std) must be computed over the full training
period (1979–2018) before training. Use `training/utils/standardization_npz_to_nc.py`
or compute them with:

```python
import xarray as xr, numpy as np

# Example: compute mean and std across all training files
# (adapt to your actual variable/file layout)
ds = xr.open_mfdataset('era5_upper_air_*.nc', combine='by_coords')
mean = ds.mean(dim='time')
std  = ds.std(dim='time')
mean.to_netcdf('pangu_s2s_1979-2018_mean.nc')
std.to_netcdf('pangu_s2s_1979-2018_std.nc')
```

---

## Expected Layout

After placing or generating the dataset, the `dataset/` directory should look like:

```
dataset/
├── 1979_0000.h5                              # 6-hourly snapshot #0 of 1979
├── 1979_0001.h5                              # 6-hourly snapshot #1
├── ...                                       # ~1460 files per year
├── 1980_0000.h5
├── ...
├── 2018_NNNN.h5                              # last file (40 years × ~1461 steps)
│
├── pangu_s2s_1979-2018_mean.nc               # Upper-air normalisation mean
├── pangu_s2s_1979-2018_std.nc                # Upper-air normalisation std
├── pangu_s2s_1979-2018_surface_mean.nc       # Surface normalisation mean
├── pangu_s2s_1979-2018_surface_std.nc        # Surface normalisation std
├── pangu_s2s_1979-2018_delta_mean.nc
├── pangu_s2s_1979-2018_delta_std.nc
├── pangu_s2s_1979-2018_surface_delta_mean.nc
├── pangu_s2s_1979-2018_surface_delta_std.nc
└── 1979-2018_mean_climatology.nc             # Monthly climatology for CRPS scoring
```

**Total size:** ~195 GB (7313 files)

---

## Variable Groups (from `training/config/exp1_dsai.yaml`)

| Group | Variables | Notes |
|---|---|---|
| Upper air | temperature, u/v wind, specific humidity, geopotential | 17 pressure levels |
| Surface | 2m temp, 10m u/v wind, MSLP, surface pressure | |
| Diagnostic | total precipitation (24hr), top LW radiation | |
| Land | soil water layer 1, soil temp layer 1, skin temp | Masked over ocean |
| Ocean | sea surface temperature | Masked over land |
| Constant boundary | land-sea mask, surface geopotential | Static fields |
| Varying boundary | TOA incident solar radiation | Changes with time/season |

**Pressure levels:** 5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa (17 levels)

**Grid:** 1°×1°, 180×360 points, global

**Temporal coverage:** 1979–2018, 6-hourly (4 snapshots/day)

---

## Minimum Dataset for a Benchmark Run

The benchmark scripts use `train_year_start: 1979` and `train_year_end: 1980`
(2 years = ~2920 HDF5 files = ~10 GB). This is sufficient to reproduce the
throughput numbers in the report.

To run with only 1979–1980 data, set in `training/config/exp1_dsai.yaml`:
```yaml
train_year_start: 1979
train_year_end: 1980
val_year_start: 1980
val_year_end: 1980
```
