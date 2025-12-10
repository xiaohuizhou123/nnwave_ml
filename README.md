# WW3 → Air-Entrainment Velocity (Va) Machine Learning Pipeline

This repository contains an end-to-end pipeline to build a pointwise machine-learning parameterization of air-entrainment velocity (`wcm` / `V_a`) from WAVEWATCH III (WW3) simulations.

The workflow:

1. Convert yearly/monthly WW3 NetCDF outputs to chunked Zarr stores.
2. Build ocean/finite-data masks.
3. Compute global feature/target scalers and time splits (80/10/10).
4. Train a pointwise MLP (4×512 GELU) using PyTorch (single-GPU or DDP).
5. Run distributed prediction back to global gridded NetCDF (per year).

## Scripts

All main scripts live in `scripts/`:

### 1. `01_convert_nc_to_zarr.py`

Convert yearly (or monthly) WW3 NetCDF files into chunked Zarr stores with optional Dask parallelism.

Example:

```bash
python scripts/01_convert_nc_to_zarr.py \
  --in /path/to/ww3.1981.nc \
  --out-root /path/to/zarr_root \
  --vars hs fp uwnd vwnd dpt wcm \
  --time-chunk 32 --lat-chunk 256 --lon-chunk 256 \
  --use-dask --n-workers 8 --threads-per-worker 2 --engine netcdf4 --overwrite

