#!/usr/bin/env python3
import argparse, os, shutil
from pathlib import Path
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

def parse():
    ap = argparse.ArgumentParser("Build ocean/finite-data mask from a yearly Zarr store.")
    ap.add_argument("--zarr", required=True, help="Input Zarr store (one year), e.g. /.../ww3.1981.zarr")
    ap.add_argument("--outdir", required=True, help="Directory to write mask (creates <stem>_mask.zarr)")
    ap.add_argument("--min-depth", type=float, default=1.0, help="Minimum depth considered ocean")
    ap.add_argument("--per-time", action="store_true",
                    help="If set, output a 3D mask(time, lat, lon). Otherwise 2D mask(lat, lon)")
    ap.add_argument("--strict", action="store_true",
                    help="Require ALL variables finite at ALL times (else OR across vars and ANY in time)")
    # Dask options
    ap.add_argument("--use-dask", action="store_true", help="Start a local Dask cluster")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--threads-per-worker", type=int, default=1)
    # Zarr writing
    ap.add_argument("--zarr-format", type=int, choices=[2,3], default=3, help="Zarr format to write (2 or 3)")
    ap.add_argument("--no-consolidate", action="store_true", help="Do not consolidate metadata after write")
    return ap.parse_args()

def _maybe_client(args):
    if not args.use_dask:
        return None
    try:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=args.n_workers,
                               threads_per_worker=args.threads_per_worker,
                               processes=True, silence_logs="WARNING")
        client = Client(cluster)
        print(f"[DASK] LocalCluster: {args.n_workers}×{args.threads_per_worker}")
        return client
    except Exception as e:
        print(f"[WARN] Dask setup failed ({e}); continuing without Dask.")
        return None

def open_zarr_any(p):
    try:  return xr.open_zarr(p, consolidated=True)
    except Exception: return xr.open_zarr(p, consolidated=False)

def pick(ds, names):
    for n in names:
        if n in ds: return n
    return None

def infer_dims(ds):
    t="time"
    y="lat" if "lat" in ds.dims else ("latitude" if "latitude" in ds.dims else "y")
    x="lon" if "lon" in ds.dims else ("longitude" if "longitude" in ds.dims else "x")
    return t,y,x

def write_mask(da, out_store, zarr_format=3, consolidate=True):
    """Write mask DataArray -> Zarr."""
    if Path(out_store).exists():
        shutil.rmtree(out_store, ignore_errors=True)
    ds_out = da.to_dataset(name="ocean_mask")
    if zarr_format == 2:
        ds_out.to_zarr(out_store, mode="w", consolidated=consolidate)
    else:
        # xarray uses `zarr_format` kw since 2024. If your xarray is older, omit and it will default.
        try:
            ds_out.to_zarr(out_store, mode="w", zarr_format=3)
        except TypeError:
            # older xarray that doesn’t expose zarr_format: just write; it may default to v3 if zarr>=3
            ds_out.to_zarr(out_store, mode="w")
        if consolidate:
            try:
                import zarr
                zarr.consolidate_metadata(out_store)
            except Exception as e:
                print(f"[WARN] consolidate_metadata failed: {e}")
    print(f"[OK] wrote {out_store}")

def main():
    args = parse()
    # be nice on HPC
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v,"1")

    client = _maybe_client(args)

    ds = open_zarr_any(args.zarr)
    t,y,x = infer_dims(ds)

    # depth variable
    depth_name = pick(ds, ["dpt","depth","bathymetry"])
    assert depth_name is not None, "Need depth variable (dpt/depth/bathymetry)."
    # candidate inputs (any subset present is fine)
    cand = [v for v in ["hs","U10","u10","uwnd","vwnd","wdir","fp","tp","wcm"] if v in ds.variables]
    if not cand:
        raise SystemExit("No candidate variables found to assess finiteness.")

    # ocean mask from depth
    depth2d = ds[depth_name]
    if depth2d.ndim == 3:  # (time, y, x) → take first slice
        depth2d = depth2d.isel({t: 0})
    ocean2d = (depth2d > args.min_depth)

    # finite maps (dask-parallelized)
    finite_maps = [xr.apply_ufunc(np.isfinite, ds[v], dask="parallelized") for v in cand]
    finite_stack = xr.concat(finite_maps, dim="vars")

    if args.per_time:
        # Require finite across variables; reduce only across "vars"
        finite_any = finite_stack.all(dim="vars") if args.strict else finite_stack.any(dim="vars")
        # Combine with ocean mask (broadcast to time)
        mask3d = (finite_any & ocean2d).astype(np.uint8).transpose(t, y, x)
        da_mask = mask3d
    else:
        # 2D mask: reduce across vars and time
        if args.strict:
            ok2d = finite_stack.all(dim="vars").all(dim=t)
        else:
            ok2d = finite_stack.any(dim="vars").any(dim=t)
        mask2d = (ocean2d & ok2d).astype(np.uint8)
        da_mask = mask2d

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_store = str(outdir / (Path(args.zarr).stem + "_mask.zarr"))

    # Compute & write
    with ProgressBar():
        da_mask = da_mask.compute()  # drain dask graph
        write_mask(da_mask, out_store,
                   zarr_format=args.zarr_format,
                   consolidate=(not args.no_consolidate))

    if client is not None:
        try: client.close()
        except Exception: pass

if __name__ == "__main__":
    dask.config.set({"array.slicing.split_large_chunks": True})
    main()

