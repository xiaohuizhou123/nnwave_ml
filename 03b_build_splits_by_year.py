#!/usr/bin/env python3
import argparse, json, os, glob
from pathlib import Path
import xarray as xr

def open_zarr_any(p):
    try:  return xr.open_zarr(p, consolidated=True)
    except Exception: return xr.open_zarr(p, consolidated=False)

def parse_years(s: str):
    # e.g. "1980-1985,1987,1990-1992"
    years=set()
    for part in s.split(","):
        part=part.strip()
        if not part: continue
        if "-" in part:
            a,b=part.split("-")
            years.update(range(int(a), int(b)+1))
        else:
            years.add(int(part))
    return years

def main():
    ap = argparse.ArgumentParser("Write explicit by-year splits into feature_scaler.json")
    ap.add_argument("--zarr_glob", required=True)
    ap.add_argument("--scaler_json_in", required=True)
    ap.add_argument("--scaler_json_out", required=True)
    ap.add_argument("--train_years", required=True, help='e.g. "1980-2010"')
    ap.add_argument("--val_years",   required=True, help='e.g. "2011-2015"')
    ap.add_argument("--test_years",  required=True, help='e.g. "2016-2022"')
    args = ap.parse_args()

    trainY = parse_years(args.train_years)
    valY   = parse_years(args.val_years)
    testY  = parse_years(args.test_years)

    with open(args.scaler_json_in, "r") as f:
        sc = json.load(f)

    by_store = {}
    paths = sorted(glob.glob(os.path.expanduser(args.zarr_glob)))
    for p in paths:
        base = os.path.basename(p)          # e.g. ww3.1987.zarr
        stem = Path(p).stem                  # ww3.1987
        year = int(stem.split(".")[-1])     # 1987
        ds = open_zarr_any(p)
        T = ds.sizes["time"]
        if year in trainY:
            by_store[base] = {"train": list(range(T)), "val": [], "test": []}
        elif year in valY:
            by_store[base] = {"train": [], "val": list(range(T)), "test": []}
        elif year in testY:
            by_store[base] = {"train": [], "val": [], "test": list(range(T))}
        else:
            # If a year isn't listed anywhere, exclude it from training entirely
            by_store[base] = {"train": [], "val": [], "test": []}
        ds.close()

    sc.setdefault("splits", {})
    sc["splits"]["by_store"] = by_store
    sc["splits"]["fractions"] = {"train": None, "val": None, "test": None}
    sc["splits"]["seed"] = None

    Path(os.path.dirname(args.scaler_json_out)).mkdir(parents=True, exist_ok=True)
    with open(args.scaler_json_out, "w") as f:
        json.dump(sc, f, indent=2)
    print(f"[OK] wrote explicit year splits to {args.scaler_json_out}")

if __name__ == "__main__":
    main()

