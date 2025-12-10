#!/usr/bin/env python3
"""
Convert yearly (or monthly) WW3 NetCDF files into chunked Zarr stores with optional Dask parallelism.

Examples
--------
python convert_nc_to_zarr.py \
  --in /home1/10677/xiaohuiz3021/scratch/Simulations/Rundir/ww3.2021.nc \
  --out-root /home1/10677/xiaohuiz3021/scratch/NNWave/data/zarr \
  --vars hs fp uwnd vwnd dpt wcm \
  --time-chunk 32 --lat-chunk 256 --lon-chunk 256 \
  --use-dask --n-workers 8 --threads-per-worker 2 --engine netcdf4 --overwrite
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import xarray as xr


def parse_args():
    ap = argparse.ArgumentParser(description="NetCDF ➜ Zarr converter with optional Dask parallelism.")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="Input NetCDF files (yearly/monthly).")
    ap.add_argument("--out-root", required=True,
                    help="Directory to write Zarr outputs (one subfolder per input).")
    ap.add_argument("--vars", nargs="+", required=True,
                    help="Variables to keep (e.g., hs fp uwnd vwnd dpt wcm depth target).")
    ap.add_argument("--engine", default=None,
                    help="xarray engine: netcdf4, h5netcdf, etc. Leave blank to auto.")
    ap.add_argument("--time-chunk", type=int, default=32)
    ap.add_argument("--lat-chunk", type=int, default=256)
    ap.add_argument("--lon-chunk", type=int, default=256)
    ap.add_argument("--use-dask", action="store_true", help="Enable Dask (recommended).")
    ap.add_argument("--n-workers", type=int, default=8, help="Workers for LocalCluster.")
    ap.add_argument("--threads-per-worker", type=int, default=2, help="Threads per worker.")
    ap.add_argument("--scheduler", type=str, default="",
                    help="Connect to existing scheduler (e.g., tcp://host:8786).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing Zarr stores.")
    ap.add_argument(
        "--zarr-version",
        type=int,
        choices=[2, 3],
        default=2,
        help="Zarr format version to write. v2 supports consolidated metadata (default). v3 writes without it.",
    )
    return ap.parse_args()


def maybe_setup_dask(use_dask: bool, n_workers: int, threads_per_worker: int, scheduler: str | None):
    """Optionally start a local Dask cluster or connect to an external scheduler."""
    if not use_dask:
        return None
    try:
        if scheduler:  # connect to an existing scheduler (e.g., dask-scheduler on HPC)
            from dask.distributed import Client
            client = Client(scheduler)  # e.g., "tcp://host:8786"
            print(f"[DASK] Connected to external scheduler at {scheduler}")
            return client
        else:
            from dask.distributed import Client, LocalCluster
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                processes=True,                # 1 process per worker
                silence_logs=logging.WARNING,  # <- IMPORTANT: use logging constant
            )
            client = Client(cluster)
            print(f"[DASK] LocalCluster: {n_workers} workers × {threads_per_worker} threads")
            return client
    except Exception as e:
        print(f"[WARN] Dask setup failed ({e}); continuing without Dask.")
        return None


def build_chunks_for_dataset(ds, time_chunk: int, lat_chunk: int, lon_chunk: int) -> dict:
    """
    Build a chunk dict **only** for dims that are present in this dataset.
    Supports ('lat'|'latitude'|'y') and ('lon'|'longitude'|'x').
    """
    present = set(ds.dims)
    chunks = {}
    if "time" in present:
        chunks["time"] = time_chunk
    for cand in ("lat", "latitude", "y"):
        if cand in present:
            chunks[cand] = lat_chunk
            break
    for cand in ("lon", "longitude", "x"):
        if cand in present:
            chunks[cand] = lon_chunk
            break
    return chunks


def main():
    args = parse_args()

    # Avoid CPU over-subscription by math libs (important on HPC)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    # Optionally create/attach Dask client
    client = maybe_setup_dask(
        use_dask=args.use_dask,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        scheduler=(args.scheduler or None),
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for p in args.inputs:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] Missing input file: {p}")
            continue

        out_store = out_root / (p.stem + ".zarr")
        if out_store.exists() and not args.overwrite:
            print(f"[SKIP] {out_store} exists (use --overwrite to replace).")
            continue
        if out_store.exists() and args.overwrite:
            import shutil
            print(f"[INFO] Removing existing {out_store}")
            shutil.rmtree(out_store, ignore_errors=True)

        print(f"[INFO] Opening {p} (engine={args.engine or 'auto'})")
        ds = xr.open_dataset(p, engine=args.engine)  # open first to see dims
        print(f"[INFO] dataset dims: {dict(ds.dims)}")

        # Build chunking dict from **actual** dims in this file
        ds_chunks = build_chunks_for_dataset(
            ds,
            time_chunk=args.time_chunk,
            lat_chunk=args.lat_chunk,
            lon_chunk=args.lon_chunk,
        )

        # Reopen with chunks (or just chunk in-memory if engine refuses 'chunks' kw)
        try:
            ds.close()
            ds = xr.open_dataset(p, engine=args.engine, chunks=ds_chunks)
        except TypeError:
            ds = xr.open_dataset(p, engine=args.engine).chunk(ds_chunks)

        # Filter variables to keep
        keep = [v for v in args.vars if v in ds.variables]
        missing = [v for v in args.vars if v not in ds.variables]
        if missing:
            print(f"[WARN] Missing vars in {p.name}: {missing}")
        if not keep:
            print("[ERROR] None of the requested variables found; skipping.")
            ds.close()
            continue

        ds_sel = ds[keep].chunk(ds_chunks)

        # Write Zarr (compute=True triggers Dask graph execution now)
        wrote = False
        try:
            ds_sel.to_zarr(out_store, mode="w", consolidated=True, compute=True)
            wrote = True
        except TypeError:
            # Older xarray: consolidated kw not supported on to_zarr
            ds_sel.to_zarr(out_store, mode="w", compute=True)
            wrote = True
            try:
                import zarr
                zarr.consolidate_metadata(str(out_store))
            except Exception as e:
                print(f"[WARN] consolidate_metadata failed: {e}")
        except Exception as e:
            print(f"[ERR] to_zarr failed for {p.name}: {e}")
            wrote = False
        finally:
            ds.close()

        if wrote:
            print(f"[OK] Wrote {out_store}")

    # Cleanly close Dask client if we started one
    if client is not None:
        try:
            client.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

