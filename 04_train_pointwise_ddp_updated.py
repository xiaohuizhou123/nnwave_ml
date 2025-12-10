#!/usr/bin/env python3
# Pointwise MLP training for WW3 → air-entrainment velocity (wcm)
# - Single-GPU or multi-node DDP (ibrun/torchrun)
# - 80/10/10 time split (uses scaler JSON "splits" if present; else computed)
# - CSV logging + live progress + resume + best/periodic checkpoints
# - Finite-depth features via dispersion solver

import os, sys, time, math, json, glob, argparse, csv, random
from pathlib import Path
import numpy as np
import xarray as xr
import torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist

# ----------------- finite-depth wave utilities -----------------
G = 9.80665
def _safe_cosh_sq(x):
    x = np.clip(x, -50.0, 50.0)             # avoid overflow
    c = np.cosh(x)
    return c * c

def solve_k_omega_h(omega, h, max_iter=30, tol=1e-10):
    omega = np.asarray(omega, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = np.maximum(omega*omega / G, 1e-8)   # deep-water initial guess
    for _ in range(max_iter):
        kh = k * h
        tanh_kh = np.tanh(np.clip(kh, -50.0, 50.0))
        sech2_kh = 1.0 / _safe_cosh_sq(kh)
        F  = G * k * tanh_kh - omega*omega
        dF = G * (tanh_kh + k * h * sech2_kh)
        step = F / (dF + 1e-16)
        k_next = np.maximum(k - step, 1e-12)
        if np.all(np.abs(step) < tol*np.maximum(k,1.0)):
            k = k_next; break
        k = k_next
    return k.astype(np.float32)

def cp_from_fp_tp(fp=None, tp=None, depth=None):
    if fp is None and tp is None:
        raise ValueError("Need fp or tp to compute cp.")
    if fp is not None:
        omega = 2.0*np.pi*np.clip(fp, 1e-6, None)
    else:
        omega = 2.0*np.pi/np.clip(tp, 1e-3, None)
    k = solve_k_omega_h(omega, depth)
    cp = omega / np.maximum(k, 1e-12)
    return cp.astype(np.float32)

def kp_from_fp_tp(fp=None, tp=None, depth=None):
    if fp is not None:
        omega = 2.0*np.pi*np.clip(fp, 1e-6, None)
    else:
        omega = 2.0*np.pi/np.clip(tp, 1e-3, None)
    return solve_k_omega_h(omega, depth).astype(np.float32)

# ----------------- utils -----------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def pick(ds, names):
    for n in names:
        if n in ds.variables: return n
    return None

def dims(ds):
    t = "time"
    y = "lat" if "lat" in ds.dims else ("latitude" if "latitude" in ds.dims else "y")
    x = "lon" if "lon" in ds.dims else ("longitude" if "longitude" in ds.dims else "x")
    return t, y, x

def time_split_indices(T, split=(0.8, 0.1, 0.1)):
    a,b,c = split
    i_train = int(T * a)
    i_val   = int(T * (a + b))
    return i_train, i_val

def open_zarr_any(p):
    try:  return xr.open_zarr(p, consolidated=True)
    except Exception: return xr.open_zarr(p, consolidated=False)

def open_mask_any(p):
    try:
        m = xr.open_zarr(p, consolidated=True)["ocean_mask"]
    except Exception:
        m = xr.open_zarr(p, consolidated=False)["ocean_mask"]
    return m

# ----------------- DDP init (works for ibrun or torchrun) -----------------
def ddp_init(force_cpu: bool = False):
    """
    Returns: (use_ddp, rank, world, local_rank, device)
    - Single GPU: returns device cuda:0 (or cpu) and use_ddp=False
    - ibrun (MPI/PMI): uses PMI_RANK/PMI_SIZE and LOCAL_RANK if present
    - torchrun: uses RANK/WORLD_SIZE/LOCAL_RANK
    """
    if force_cpu or not torch.cuda.is_available():
        return False, 0, 1, 0, torch.device("cpu")

    rank = os.environ.get("RANK"); world = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")

    # fallback to PMI (ibrun)
    if rank is None or world is None:
        pr = os.environ.get("PMI_RANK"); ps = os.environ.get("PMI_SIZE")
        if pr is not None and ps is not None:
            rank = pr; world = ps
            local_rank = os.environ.get("I_MPI_LOCAL_RANK") or os.environ.get("PMI_LOCAL_RANK") or "0"

    # single-process
    if rank is None or world is None or local_rank is None:
        torch.cuda.set_device(0)
        return False, 0, 1, 0, torch.device("cuda", 0)

    rank = int(rank); world = int(world); local_rank = int(local_rank)
    ndev = torch.cuda.device_count()
    if ndev < 1:
        raise RuntimeError("No CUDA devices visible.")
    if not (0 <= local_rank < ndev):
        local_rank = local_rank % ndev

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world)
    return True, rank, world, local_rank, torch.device("cuda", local_rank)

def is_main(use_ddp, rank): return (not use_ddp) or (rank==0)

# ----------------- Dataset -----------------
class PointwiseDataset(IterableDataset):
    """
    Random point sampler over (time, y, x). Mode: 'train' or 'val' or 'predict'.
    Uses scaler-normalized 7 features: [hs, U10, cosw, sinw, wave_age, steepness, depth] and target y (wcm).
    """
    def __init__(self, zarr_paths, mask_root, scaler_json, mode="train",
                 batch_points=131072, seed=1234, rank=0):
        super().__init__()
        self.paths = sorted(zarr_paths)
        self.mask_root = Path(mask_root)
        self.batch_points = int(batch_points)
        self.seed = int(seed) + 1009*int(rank)
        self.mode = mode

        # scaler & optional precomputed splits
        s = json.load(open(scaler_json, "r"))
        self.mean = np.array(s["mean"], np.float32)
        self.std  = np.array(s["std"],  np.float32); self.std[self.std==0]=1.0
        self.y_mu = np.float32(s.get("y_mean", 0.0))
        self.y_sd = np.float32(s.get("y_std", 1.0)) if s.get("y_std", 0.0)!=0 else np.float32(1.0)
        self.target = s.get("target", "wcm")
        self.splits = s.get("splits", {}).get("by_store", {})  # optional

        self._opened = None  # lazily open in __iter__

    def _open_all(self):
        opened = []
        for p in self.paths:
            base = os.path.basename(p)
            ds = open_zarr_any(p)
            t,y,x = dims(ds); T = ds.sizes[t]
            # pick time indices for this split
            if base in self.splits:
                # expected structure: {"train":[...], "val":[...], "predict":[...]} or separate key
                split_map = self.splits[base]
                if isinstance(split_map, dict):    # new format
                    idx = split_map.get(self.mode, [])
                else:                               # legacy list per split_key in older script
                    idx = split_map
            else:
                itrain, ival = time_split_indices(T, (0.8,0.1,0.1))
                if self.mode == "train":   idx = np.arange(0, max(itrain,1))
                elif self.mode == "val":   idx = np.arange(itrain, max(ival,itrain+1))
                else:                      idx = np.arange(ival, T)

            if len(idx)==0: 
                continue

            # ocean mask (2D preferred)
            mpath = self.mask_root / (Path(p).stem + "_mask.zarr")
            m = open_mask_any(str(mpath))
            if "time" in m.dims:
                ocean2d = m.any(dim="time")
            else:
                ocean2d = m
            ocean2d = ocean2d.load().values.astype(bool)
            yy, xx = np.where(ocean2d)
            if yy.size == 0:
                continue

            opened.append((base, ds, np.array(idx, dtype=np.int64), (yy, xx), (t,y,x)))
        if not opened:
            raise RuntimeError("No datasets available for split '%s'." % self.mode)
        return opened

    def __iter__(self):
        if self._opened is None:
            self._opened = self._open_all()
        rng = np.random.default_rng(self.seed + (os.getpid() % 99991))

        while True:
            Xb = np.empty((self.batch_points, 7), dtype=np.float32)
            yb = np.empty((self.batch_points,), dtype=np.float32)
            filled = 0

            while filled < self.batch_points:
                base, ds, tids, (yy,xx), (t,y,x) = self._opened[rng.integers(0, len(self._opened))]
                ti = int(tids[rng.integers(0, tids.size)])
                # sample many points at once
                need = self.batch_points - filled
                take = min(need, 16384)
                sel = rng.integers(0, yy.size, size=take)
                ys = yy[sel]; xs = xx[sel]

                # pull 1D slices at a single time
                try:
                    def get1(var):
                        return ds[var].isel({t: ti, y: xr.DataArray(ys, dims="z"), x: xr.DataArray(xs, dims="z")}).load().values.astype(np.float32)

                    hsN  = pick(ds, ["hs","Hs","SWH"]);       dN   = pick(ds, ["dpt","depth","bathymetry"])
                    uwN  = pick(ds, ["uwnd","uw10","u10","u_wind"]); vwN = pick(ds, ["vwnd","vw10","v10","v_wind"])
                    u10N = pick(ds, ["U10","u10"])
                    wdN  = pick(ds, ["wdir","wind_direction"])
                    fpN  = pick(ds, ["fp","fpeak","peak_frequency"]); tpN = pick(ds, ["tp","tpeak","peak_period"])
                    yN   = self.target if self.target in ds.variables else pick(ds, [self.target,"wcm","target"])
                    # required vars
                    if (hsN is None) or (dN is None) or (yN is None) or ((u10N is None) and (uwN is None or vwN is None)) or ((fpN is None) and (tpN is None)):
                        continue

                    hs = get1(hsN)
                    if u10N: U10 = get1(u10N)
                    else:    U10 = np.sqrt(np.maximum(get1(uwN)**2 + get1(vwN)**2, 0.0)).astype(np.float32)

                    if wdN:
                        wdir = get1(wdN) * (np.pi/180.0)
                        cosw, sinw = np.cos(wdir), np.sin(wdir)
                    else:
                        # derive from uwnd/vwnd
                        if u10N: uw = get1(uwN); vw = get1(vwN)
                        else:    uw = get1(uwN);  vw = get1(vwN)
                        wdir = np.arctan2(vw, uw).astype(np.float32)
                        cosw, sinw = np.cos(wdir), np.sin(wdir)

                    # depth may be 2D (y,x) or 3D (time,y,x)
                    depth_v = ds[dN]
                    if depth_v.ndim == 2:
                        depth = depth_v.isel({y: xr.DataArray(ys, dims="z"), x: xr.DataArray(xs, dims="z")}).load().values.astype(np.float32)
                    else:
                        depth = depth_v.isel({t: ti, y: xr.DataArray(ys, dims="z"), x: xr.DataArray(xs, dims="z")}).load().values.astype(np.float32)

                    if fpN is not None:
                        fp = get1(fpN); tp = None
                    else:
                        tp = get1(tpN); fp = None

                    cp = cp_from_fp_tp(fp=fp, tp=tp, depth=depth)
                    kp = kp_from_fp_tp(fp=fp, tp=tp, depth=depth)
                    wave_age = (cp / np.clip(U10, 0.5, None)).astype(np.float32)
                    steep    = (kp * hs / 2.0).astype(np.float32)

                    yv = get1(yN)

                    X = np.stack([hs, U10, cosw, sinw, wave_age, steep, depth], axis=-1)
                    finite = np.isfinite(X).all(axis=-1) & np.isfinite(yv)
                    if not finite.any():
                        continue
                    X = X[finite]; yv = yv[finite]

                    take2 = min(need, X.shape[0])
                    X = X[:take2]; yv = yv[:take2]

                    # normalize
                    X = (X - self.mean) / self.std
                    yv = (yv - self.y_mu) / self.y_sd

                    Xb[filled:filled+take2] = X
                    yb[filled:filled+take2] = yv
                    filled += take2
                except Exception:
                    continue

            yield torch.from_numpy(Xb), torch.from_numpy(yb)

# ----------------- Model -----------------
class MLP(nn.Module):
    def __init__(self, in_dim=7, hidden=512, depth=4, dropout=0.1):
        super().__init__()
        layers=[]; d=in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ----------------- Train -----------------
def train(cfg):
    # prevent BLAS oversubscription
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v,"1")

    # quick probe
    print("RANK", os.environ.get("RANK"), "WORLD_SIZE", os.environ.get("WORLD_SIZE"),
          "LOCAL_RANK", os.environ.get("LOCAL_RANK"), "PMI_RANK", os.environ.get("PMI_RANK"))
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available() =", torch.cuda.is_available(),
          "cuda.count =", torch.cuda.device_count())
    sys.stdout.flush()

    use_ddp, rank, world, local_rank, device = ddp_init(force_cpu=False)
    set_seed(cfg.seed + (rank if use_ddp else 0) * 1009)

    # expand globs to list of zarr stores
    zlist = []
    for g in cfg.zarr_glob.split():
        zlist.extend(sorted(glob.glob(os.path.expanduser(g))))
    if not zlist:
        raise SystemExit(f"No zarr stores matched: {cfg.zarr_glob}")

    # datasets/loaders
    train_ds = PointwiseDataset(zarr_paths=zlist, mask_root=cfg.mask_root, scaler_json=cfg.scaler_json,
                                mode="train", batch_points=cfg.batch_points, seed=cfg.seed, rank=rank)
    val_ds   = PointwiseDataset(zarr_paths=zlist, mask_root=cfg.mask_root, scaler_json=cfg.scaler_json,
                                mode="val",   batch_points=max(32768, cfg.batch_points//4), seed=cfg.seed+123, rank=rank)

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=max(1, cfg.num_workers//2), pin_memory=True)

    model = MLP(in_dim=7, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler() if device.type=="cuda" else None

    os.makedirs(cfg.save_dir, exist_ok=True)
    csv_writer = None; csv_fh = None
    if is_main(use_ddp, rank) and cfg.log_csv:
        os.makedirs(Path(cfg.log_csv).parent, exist_ok=True)
        csv_fh = open(cfg.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch","step","train_mse","val_mse","steps_per_sec","samples_per_sec_per_gpu"])

    def log_row(epoch, step, tr, va, sps, sppg):
        if csv_writer:
            csv_writer.writerow([epoch, step, tr, va, sps, sppg]); csv_fh.flush()

    # ---- resume ----
    start_epoch = 1; best = float("inf")
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu")
        (model.module if use_ddp else model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if scaler_amp and ("scaler" in ckpt):
            scaler_amp.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best", float("inf")))
        if is_main(use_ddp, rank):
            print(f"[RESUME] from {cfg.resume} at epoch {start_epoch} (best={best:.6f})", flush=True)

    for ep in range(start_epoch, cfg.epochs+1):
        model.train(); tr_loss=0.0; n_seen=0
        ep_t0 = time.perf_counter()
        it_train = iter(train_loader)
        step_t0 = time.perf_counter()

        for step_idx in range(1, cfg.train_steps_per_epoch+1):
            Xb,yb = next(it_train)
            bs = Xb.shape[0]
            X = Xb.to(device, non_blocking=True); y = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    pred = model(X); loss = loss_fn(pred, y)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler_amp.step(opt); scaler_amp.update()
            else:
                pred = model(X); loss = loss_fn(pred, y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            tr_loss += loss.item() * bs; n_seen += bs

            # progress
            if is_main(use_ddp, rank) and (step_idx % cfg.log_every == 0 or step_idx == cfg.train_steps_per_epoch):
                now = time.perf_counter(); dt = now - step_t0; step_t0 = now
                steps_per_sec = (cfg.log_every if step_idx % cfg.log_every == 0 else 1.0) / max(dt,1e-9)
                sppg = int((bs * (cfg.log_every if step_idx % cfg.log_every == 0 else 1)) / max(dt,1e-9))
                pct = 100.0 * step_idx / cfg.train_steps_per_epoch
                print(f"[ep {ep:03d} {step_idx:04d}/{cfg.train_steps_per_epoch}] "
                      f"{pct:5.1f}% • {dt:.3f}s/print • {steps_per_sec:.2f} steps/s • "
                      f"{sppg} samples/s/gpu • train_mse(cur)={loss.item():.6f}", flush=True)

        tr_mse = tr_loss / max(n_seen,1)

        # validation
        model.eval(); va_loss=0.0; m_seen=0
        it_val = iter(val_loader)
        with torch.no_grad():
            for _ in range(cfg.val_steps):
                Xv,yv = next(it_val)
                X = Xv.to(device, non_blocking=True); y = yv.to(device, non_blocking=True)
                if scaler_amp:
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        pv = model(X); lv = loss_fn(pv, y)
                else:
                    pv = model(X); lv = loss_fn(pv, y)
                va_loss += lv.item() * X.shape[0]; m_seen += X.shape[0]
        va_mse = va_loss / max(m_seen,1)

        # all-reduce for DDP
        if use_ddp:
            t = torch.tensor([tr_mse, va_mse], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM); t /= dist.get_world_size()
            tr_mse, va_mse = t.tolist()

        # epoch summary + CSV + checkpoints
        if is_main(use_ddp, rank):
            elapsed = max(time.perf_counter() - ep_t0, 1e-9)
            steps_per_sec = cfg.train_steps_per_epoch / elapsed
            avg_bs = n_seen / max(cfg.train_steps_per_epoch,1)
            sppg = int((avg_bs * cfg.train_steps_per_epoch) / elapsed)
            print(f"[ep {ep:03d}] train_mse={tr_mse:.6f}  val_mse={va_mse:.6f}  "
                  f"{steps_per_sec:.2f} steps/s  {sppg} samples/s/gpu", flush=True)
            log_row(ep, cfg.train_steps_per_epoch, tr_mse, va_mse, steps_per_sec, sppg)

            # save "last" (for resume) every epoch
            last_path = os.path.join(cfg.save_dir, "last.pt")
            torch.save({
                "model": (model.module if use_ddp else model).state_dict(),
                "opt": opt.state_dict(),
                "scaler": (scaler_amp.state_dict() if scaler_amp else None),
                "epoch": ep,
                "best": best,
            }, last_path)

            # save best & periodic
            if va_mse < best:
                best = va_mse
                torch.save((model.module if use_ddp else model).state_dict(),
                           os.path.join(cfg.save_dir, "model.pt"))
            if (ep % cfg.save_every) == 0:
                torch.save((model.module if use_ddp else model).state_dict(),
                           os.path.join(cfg.save_dir, f"model_ep{ep}.pt"))

    if csv_fh: csv_fh.close()
    if use_ddp:
        dist.destroy_process_group()

# ----------------- CLI -----------------
def parse():
    ap = argparse.ArgumentParser("Pointwise MLP on 7 features (finite-depth); single-GPU or multi-node DDP.")
    ap.add_argument("--zarr_glob", required=True, help='One or many globs, space-separated, e.g. "/path/ww3.*.zarr"')
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--scaler_json", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train_steps_per_epoch", type=int, default=400)
    ap.add_argument("--val_steps", type=int, default=40)
    ap.add_argument("--batch_points", type=int, default=131072)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_dir", default="./models_pointwise")
    ap.add_argument("--log_csv", default="", help="Write CSV metrics here")
    ap.add_argument("--log_every", type=int, default=20, help="Print every N steps")
    ap.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    ap.add_argument("--resume", default="", help="Path to last.pt to resume")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pull_size", type=int, default=65536, help="points pulled per I/O gather")
    ap.add_argument("--shard_by_store", action="store_true", help="each rank takes a disjoint set of stores")
    return ap.parse_args()

if __name__ == "__main__":
    cfg = parse()
    train(cfg)

