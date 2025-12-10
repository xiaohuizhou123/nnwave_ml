#!/usr/bin/env python3
# Predict pointwise air-entrainment (wcm) and write NetCDF per Zarr store,
# including selected feature fields (hs, cp, U10).
#
# Usage (single store):
#   python 05_predict_pointwise_dist.py \
#     --zarr_glob "/path/zarr/ww3.1981.zarr" \
#     --mask_root "/path/prep" \
#     --scaler_json "/path/prep/feature_scaler.json" \
#     --model "/path/models_pointwise_full/model.pt" \
#     --out_dir "/path/pred_nc" --save_features hs cp U10
#
# Multi-node (Vista GH, 8 nodes, 1 GPU/node):
#   ibrun -np 8 python 05_predict_pointwise_dist.py \
#     --zarr_glob "/path/zarr/ww3.*.zarr" \
#     --mask_root "/path/prep" \
#     --scaler_json "/path/prep/feature_scaler.json" \
#     --model "/path/models_pointwise_full/model.pt" \
#     --out_dir "/path/pred_nc" \
#     --save_features hs cp U10 \
#     --time_block 16 --batch_points 131072 --shard_by_store

import os, sys, glob, json, argparse, time
from pathlib import Path
import numpy as np
import xarray as xr
import torch, torch.nn as nn
import torch.distributed as dist

# ---------- finite-depth dispersion (same as trainer) ----------
G = 9.80665
def _safe_cosh_sq(x):
    x = np.clip(x, -50.0, 50.0)
    c = np.cosh(x)
    return c*c

def solve_k_omega_h(omega, h, max_iter=30, tol=1e-10):
    omega = np.asarray(omega, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = np.maximum(omega*omega / G, 1e-8)
    for _ in range(max_iter):
        kh = np.clip(k*h, -50.0, 50.0)
        tanh_kh = np.tanh(kh)
        sech2_kh = 1.0 / _safe_cosh_sq(kh)
        F  = G*k*tanh_kh - omega*omega
        dF = G*(tanh_kh + k*h*sech2_kh)
        step = F / (dF + 1e-16)
        k = np.maximum(k - step, 1e-12)
        if np.all(np.abs(step) < tol*np.maximum(k,1.0)): break
    return k.astype(np.float32)

def cp_from_fp_tp(fp=None, tp=None, depth=None):
    if fp is None and tp is None:
        raise ValueError("Need fp or tp to compute cp.")
    omega = 2*np.pi*(np.clip(fp, 1e-6, None) if fp is not None else 1/np.clip(tp, 1e-3, None))
    k = solve_k_omega_h(omega, depth)
    return (omega/np.maximum(k,1e-12)).astype(np.float32)

def kp_from_fp_tp(fp=None, tp=None, depth=None):
    omega = 2*np.pi*(np.clip(fp, 1e-6, None) if fp is not None else 1/np.clip(tp, 1e-3, None))
    return solve_k_omega_h(omega, depth).astype(np.float32)

# ---------- model (same as trainer) ----------
class MLP(nn.Module):
    def __init__(self, in_dim=7, hidden=512, depth=4, dropout=0.1):
        super().__init__()
        layers=[]; d=in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d=hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: [N,7]
        return self.net(x).squeeze(-1)

# ---------- helpers ----------
def pick(ds, names):
    for n in names:
        if n in ds.variables: return n
    return None

def dims(ds):
    t="time"
    y="lat" if "lat" in ds.dims else ("latitude" if "latitude" in ds.dims else "y")
    x="lon" if "lon" in ds.dims else ("longitude" if "longitude" in ds.dims else "x")
    return t,y,x

def open_zarr_any(p):
    try:  return xr.open_zarr(p, consolidated=True)
    except Exception: return xr.open_zarr(p, consolidated=False)

def open_mask_any(p):
    try:  return xr.open_zarr(p, consolidated=True)["ocean_mask"]
    except Exception: return xr.open_zarr(p, consolidated=False)["ocean_mask"]

def ddp_init():
    if not torch.cuda.is_available():
        return False, 0, 1, 0, torch.device("cpu")
    rank=os.environ.get("RANK"); world=os.environ.get("WORLD_SIZE"); local_rank=os.environ.get("LOCAL_RANK")
    if rank is None or world is None:
        pr=os.environ.get("PMI_RANK"); ps=os.environ.get("PMI_SIZE")
        if pr is not None and ps is not None:
            rank, world = pr, ps
            local_rank = os.environ.get("I_MPI_LOCAL_RANK") or os.environ.get("PMI_LOCAL_RANK") or "0"
    if rank is None or world is None or local_rank is None:
        torch.cuda.set_device(0)
        return False, 0, 1, 0, torch.device("cuda",0)
    rank, world, local_rank = int(rank), int(world), int(local_rank)
    ndev = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % max(1,ndev))
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world)
    return True, rank, world, local_rank, torch.device("cuda", local_rank%max(1,ndev))

def is_main(use_ddp, rank): return (not use_ddp) or (rank==0)

def time_split_indices(T, split=(0.8,0.1,0.1)):
    a,b,_=split
    i_tr=int(T*a); i_val=int(T*(a+b))
    return i_tr, i_val

# ---------- main predict ----------
def predict_store(store_path, mask_root, model, device, scaler, save_features, out_dir,
                  mode, time_block, batch_points, rank=0, world=1):
    """
    Write one NetCDF per store with predicted wcm and requested features.
    Shard time by rank if world>1 and --shard_by_store not used.
    """
    ds = open_zarr_any(store_path)
    t,y,x = dims(ds)
    T,H,W = ds.sizes[t], ds.sizes[y], ds.sizes[x]

    # choose time indices for "predict" part
    base = os.path.basename(store_path)
    if "splits" in scaler and "by_store" in scaler["splits"] and base in scaler["splits"]["by_store"]:
        tids = np.array(scaler["splits"]["by_store"][base].get("predict", []), dtype=np.int64)
    else:
        i_tr, i_val = time_split_indices(T)
        tids = np.arange(i_val, T, dtype=np.int64)  # 10% tail by default

    if tids.size == 0:
        print(f"[rank {rank}] {base}: no predict timesteps; skip.")
        return

    # load/create 2D ocean mask
    mpath = Path(mask_root)/ (Path(store_path).stem + "_mask.zarr")
    m = open_mask_any(str(mpath))
    ocean2d = m.any(dim="time") if "time" in m.dims else m
    ocean2d = ocean2d.load().values.astype(bool)
    ocean_idx = np.where(ocean2d)  # tuple (yy, xx)
    n_ocean = ocean_idx[0].size
    if n_ocean == 0:
        print(f"[rank {rank}] {base}: mask empty; skip.")
        return

    # shard time across ranks (if NOT sharding by store):
    # here we just stride by world so each rank gets ~1/world timesteps
    if world > 1:
        tids = tids[rank::world]
        if tids.size == 0:
            print(f"[rank {rank}] {base}: no local timesteps after time sharding; skip.")
            return

    # prepare output arrays (lazy, chunk by time)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / (Path(store_path).stem + "_predict.nc")
    tmp_nc = out_dir / (Path(store_path).stem + f"_predict.rank{rank}.tmp.nc")

    # coordinate arrays
    coords = {
        t: ds[t].values,
        y: ds[y].values,
        x: ds[x].values,
    }

    # alloc on demand: we’ll build per-time Dataset and append by concat at end (rank 0),
    # but to keep memory low, we write a per-rank NetCDF and let rank 0 merge.
    # Here we just write the local time-slab file (tmp_nc), then rank 0 gathers later.
    ds_template = xr.Dataset(
        coords={t: coords[t], y: coords[y], x: coords[x]},
        attrs=dict(
            title="Air-entrainment velocity prediction",
            source="Pointwise MLP (finite-depth)",
            features="hs, U10, cosw, sinw, wave_age, steepness, depth",
            extras="hs, cp, U10 stored for convenience",
        )
    )

    enc = {  # simple compression
        "wcm_pred": {"zlib": True, "complevel": 4, "chunksizes": (max(1,time_block), max(32,H//4), max(32,W//4))},
        "hs":       {"zlib": True, "complevel": 4, "chunksizes": (max(1,time_block), max(32,H//4), max(32,W//4))},
        "cp":       {"zlib": True, "complevel": 4, "chunksizes": (max(1,time_block), max(32,H//4), max(32,W//4))},
        "U10":      {"zlib": True, "complevel": 4, "chunksizes": (max(1,time_block), max(32,H//4), max(32,W//4))},
    }

    # scaler
    x_mean = np.array(scaler["mean"], np.float32)
    x_std  = np.array(scaler["std"],  np.float32); x_std[x_std==0]=1.0
    y_mu   = np.float32(scaler.get("y_mean", 0.0))
    y_sd   = np.float32(scaler.get("y_std",  1.0)) if scaler.get("y_std",0.0)!=0 else np.float32(1.0)

    # variable name resolution
    hsN  = pick(ds, ["hs","Hs","SWH"])
    uwN  = pick(ds, ["uwnd","uw10","u10","u_wind"])
    vwN  = pick(ds, ["vwnd","vw10","v10","v_wind"])
    u10N = pick(ds, ["U10","u10"])
    wdN  = pick(ds, ["wdir","wind_direction"])
    fpN  = pick(ds, ["fp","fpeak","peak_frequency"])
    tpN  = pick(ds, ["tp","tpeak","peak_period"])
    dN   = pick(ds, ["dpt","depth","bathymetry"])

    need = [hsN, dN, (u10N or (uwN and vwN)), (fpN or tpN)]
    if any(v is None for v in [hsN, dN]) or ((u10N is None) and (uwN is None or vwN is None)) or ((fpN is None) and (tpN is None)):
        print(f"[rank {rank}] {base}: variables missing; skip.", flush=True)
        return

    # model
    model.eval()

    # process in time blocks
    written = False
    with torch.no_grad():
        blocks = [tids[i:i+time_block] for i in range(0, tids.size, time_block)]
        for blk in blocks:
            Tb = len(blk)
            # allocate block slabs (NaN-filled)
            wcm_block = np.full((Tb, H, W), np.nan, dtype=np.float32)
            hs_block  = np.full((Tb, H, W), np.nan, dtype=np.float32) if ("hs" in save_features) else None
            cp_block  = np.full((Tb, H, W), np.nan, dtype=np.float32) if ("cp" in save_features) else None
            U10_block = np.full((Tb, H, W), np.nan, dtype=np.float32) if ("U10" in save_features) else None

            for j, ti in enumerate(blk):
                # gather all ocean points at this time into a big batch (bottleneck-friendly)
                ys, xs = ocean_idx
                sub = {t: int(ti), y: xr.DataArray(ys, dims="z"), x: xr.DataArray(xs, dims="z")}

                def get1(var):
                    return ds[var].isel(sub).load().values.astype(np.float32)

                hs = get1(hsN)
                if u10N:
                    U10 = get1(u10N)
                else:
                    uw = get1(uwN); vw = get1(vwN)
                    U10 = np.sqrt(np.maximum(uw*uw + vw*vw, 0.0)).astype(np.float32)

                if wdN:
                    wdir = get1(wdN)*(np.pi/180.0)
                    cosw, sinw = np.cos(wdir), np.sin(wdir)
                else:
                    if u10N:
                        # no components available—fallback to zeros for cos/sin (neutral)
                        cosw = np.ones_like(U10, dtype=np.float32)
                        sinw = np.zeros_like(U10, dtype=np.float32)
                    else:
                        uw = get1(uwN); vw = get1(vwN)
                        wdir = np.arctan2(vw, uw).astype(np.float32)
                        cosw, sinw = np.cos(wdir), np.sin(wdir)

                depth_v = ds[dN]
                if depth_v.ndim == 2:
                    depth = depth_v.isel({y: xr.DataArray(ys, dims="z"), x: xr.DataArray(xs, dims="z")}).load().values.astype(np.float32)
                else:
                    depth = depth_v.isel({t: int(ti), y: xr.DataArray(ys, dims="z"), x: xr.DataArray(xs, dims="z")}).load().values.astype(np.float32)

                if fpN is not None:
                    fp = get1(fpN); tp=None
                else:
                    tp = get1(tpN); fp=None

                cp  = cp_from_fp_tp(fp=fp, tp=tp, depth=depth)
                kp  = kp_from_fp_tp(fp=fp, tp=tp, depth=depth)
                wave_age = (cp / np.clip(U10, 0.5, None)).astype(np.float32)
                steep    = (kp * hs / 2.0).astype(np.float32)

                X = np.stack([hs, U10, cosw, sinw, wave_age, steep, depth], axis=-1)
                finite = np.isfinite(X).all(axis=-1)
                if not finite.any():
                    continue
                X = (X[finite] - x_mean) / x_std

                # infer in mini-batches if too large
                preds = np.empty((X.shape[0],), dtype=np.float32)
                B = batch_points
                for k0 in range(0, X.shape[0], B):
                    k1 = min(k0+B, X.shape[0])
                    Xt = torch.from_numpy(X[k0:k1]).to(device, non_blocking=True)
                    with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                                        dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
                        out = model(Xt).float().detach().cpu().numpy()
                    preds[k0:k1] = out

                # inverse-scale target
                preds = preds * y_sd + y_mu

                # scatter back to 2D
                arr = np.full((H,W), np.nan, dtype=np.float32)
                arr[ys, xs] = preds
                wcm_block[j] = arr

                if hs_block is not None:
                    arr_hs = np.full((H,W), np.nan, dtype=np.float32)
                    arr_hs[ys, xs] = hs
                    hs_block[j] = arr_hs
                if cp_block is not None:
                    arr_cp = np.full((H,W), np.nan, dtype=np.float32)
                    arr_cp[ys, xs] = cp
                    cp_block[j] = arr_cp
                if U10_block is not None:
                    arr_u = np.full((H,W), np.nan, dtype=np.float32)
                    arr_u[ys, xs] = U10
                    U10_block[j] = arr_u

            # build block dataset with the true time coordinates
            time_vals = coords[t][blk]
            out_vars = {
                "wcm_pred": ( (t,y,x), wcm_block )
            }
            if hs_block is not None:  out_vars["hs"]  = ( (t,y,x), hs_block )
            if cp_block is not None:  out_vars["cp"]  = ( (t,y,x), cp_block )
            if U10_block is not None: out_vars["U10"] = ( (t,y,x), U10_block )

            ds_blk = xr.Dataset(
                data_vars=out_vars,
                coords={t: time_vals, y: coords[y], x: coords[x]}
            )
            # append or write new (per-rank temp file)
            if not tmp_nc.exists():
                ds_blk.to_netcdf(tmp_nc, mode="w", encoding=enc)
            else:
                ds_blk.to_netcdf(tmp_nc, mode="a")

            written = True
            del wcm_block, hs_block, cp_block, U10_block, ds_blk

    # rank writes: tmp file ready
    if written:
        print(f"[rank {rank}] wrote {tmp_nc}", flush=True)
    else:
        print(f"[rank {rank}] nothing written for {base}", flush=True)

def main():
    ap = argparse.ArgumentParser("Distributed pointwise prediction to NetCDF (with hs, cp, U10 fields).")
    ap.add_argument("--zarr_glob", required=True, help='Glob(s) for input Zarr stores, e.g. "/path/ww3.*.zarr"')
    ap.add_argument("--mask_root", required=True, help="Directory holding *_mask.zarr stores")
    ap.add_argument("--scaler_json", required=True, help="feature_scaler.json from training")
    ap.add_argument("--model", required=True, help="model checkpoint (model.pt)")
    ap.add_argument("--out_dir", required=True, help="Output NetCDF directory")
    ap.add_argument("--save_features", nargs="*", default=["hs","cp","U10"], choices=["hs","cp","U10"],
                    help="Which feature fields to also save into NetCDF")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--time_block", type=int, default=16, help="timesteps per write block")
    ap.add_argument("--batch_points", type=int, default=131072, help="mini-batch for model forward at a time")
    ap.add_argument("--shard_by_store", action="store_true", help="each rank takes disjoint set of stores (fastest)")
    args = ap.parse_args()

    # env hygiene
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v,"1")

    use_ddp, rank, world, local_rank, device = ddp_init()
    if rank==0:
        print("[predict] args:", vars(args), flush=True)

    # model
    model = MLP(in_dim=7, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    sd = torch.load(args.model, map_location="cpu")
    # accept either pure state_dict or wrapper dict
    if isinstance(sd, dict) and all(k in sd for k in ["model","opt"]):
        model.load_state_dict(sd["model"])
    else:
        model.load_state_dict(sd)
    model.eval()

    # scaler
    scaler = json.load(open(args.scaler_json,"r"))

    # expand stores and assign to ranks
    stores=[]
    for g in args.zarr_glob.split():
        stores.extend(sorted(glob.glob(os.path.expanduser(g))))
    if not stores:
        if rank==0: print("No input zarr stores matched.", flush=True)
        if use_ddp: dist.destroy_process_group()
        return

    if args.shard_by_store and world>1:
        my_stores = [p for i,p in enumerate(stores) if (i % world) == rank]
    else:
        my_stores = stores[:]  # everyone will time-shard inside predict_store()

    for sp in my_stores:
        try:
            predict_store(
                store_path=sp, mask_root=args.mask_root, model=model, device=device,
                scaler=scaler, save_features=set(args.save_features), out_dir=args.out_dir,
                mode="predict", time_block=args.time_block, batch_points=args.batch_points,
                rank=rank, world=(1 if args.shard_by_store else world)
            )
        except Exception as e:
            print(f"[rank {rank}] ERROR on {sp}: {e}", flush=True)

    # optional: merge per-rank files into a single NetCDF per store (only if NOT shard_by_store)
    if use_ddp:
        dist.barrier()

    if (not args.shard_by_store) and (rank==0):
        # Gather rank tmp files and merge along time where needed.
        out_dir = Path(args.out_dir)
        for sp in stores:
            stem = Path(sp).stem
            tmp_parts = sorted(out_dir.glob(stem + "_predict.rank*.tmp.nc"))
            if not tmp_parts:
                continue
            # open all parts and concat over time
            dsets = [xr.open_dataset(p) for p in tmp_parts]
            # Some parts may contain disjoint time slices; concat/merge as needed
            ds_out = xr.concat(dsets, dim="time").sortby("time")
            out_nc = out_dir / (stem + "_predict.nc")
            ds_out.to_netcdf(out_nc, mode="w")
            for d in dsets: d.close()
            for p in tmp_parts: 
                try: os.remove(p)
                except: pass
            print(f"[merge] wrote {out_nc}", flush=True)

    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

