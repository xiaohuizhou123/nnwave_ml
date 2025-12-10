#!/usr/bin/env python3
"""
Parallel pointwise scaler with finite-depth physics.

- Global time-based 80/10/10 split across Zarr stores
- Parallel sampling with Dask (local cluster or external scheduler)
- Optional GPU acceleration of wave physics with CuPy (--use-cupy)
- Saves mean/std for X (7 features), y_mean/std, and per-store splits

Example (single year):
  python compute_scalers_parallel.py \
    --zarr-glob "/path/zarr/ww3.1981.zarr" \
    --mask-root "/path/prep" \
    --samples 200000 --target wcm \
    --out "/path/prep/feature_scaler_1981.json" \
    --use-dask --nworkers 16 --threads-per-worker 1

Example (1980–2022, local-node parallel):
  python compute_scalers_parallel.py \
    --zarr-glob "/path/zarr/ww3.*.zarr" \
    --mask-root "/path/prep" \
    --samples 500000 --target wcm \
    --out "/path/prep/feature_scaler.json" \
    --use-dask --nworkers 32 --threads-per-worker 1

Optional GPU physics:
  add --use-cupy  (requires cupy installed and CUDA available)
"""

import argparse, json, os, math, re
from glob import glob
from pathlib import Path
import numpy as np
import xarray as xr

# ---------- optional CuPy backend ----------
def try_import_cupy(use_cupy_flag):
    if not use_cupy_flag:
        return None
    try:
        import cupy as cp
        _ = cp.zeros((1,), dtype=cp.float32)  # smoke test
        return cp
    except Exception:
        return None

# ----- finite-depth physics (NumPy & optional CuPy) -----
G = 9.80665

def _as_f64_np(x): return np.asarray(x, dtype=np.float64)

def solve_k_omega_h_np(omega, h, max_iter=30, tol=1e-10):
    omega = _as_f64_np(omega)
    h = _as_f64_np(h)
    omega = np.clip(omega, 1e-6, None)
    h = np.clip(h, 1e-6, None)
    k = np.maximum(omega*omega / G, 1e-10)
    for _ in range(max_iter):
        kh = k * h
        tanh_kh = np.tanh(kh)
        sech2_kh = 1.0 - tanh_kh * tanh_kh
        F = G * k * tanh_kh - omega*omega
        dF = G * (tanh_kh + k * h * sech2_kh)
        dF = np.where(np.abs(dF) < 1e-16, np.sign(dF) * 1e-16, dF)
        step = F / dF
        k_next = k - step
        k_next = np.clip(k_next, 1e-12, 1e6)
        if np.all(np.abs(step) <= tol * np.maximum(k, 1.0)): k = k_next; break
        k = k_next
    return k.astype(np.float32)

def cp_from_fp_tp_np(fp=None, tp=None, depth=None):
    if fp is None and tp is None: raise ValueError("Need fp or tp")
    if depth is None: raise ValueError("Need depth")
    depth = _as_f64_np(depth)
    if fp is not None:
        omega = 2*np.pi*np.clip(_as_f64_np(fp), 1e-6, None)
    else:
        omega = 2*np.pi/np.clip(_as_f64_np(tp), 1e-3, None)
    k = solve_k_omega_h_np(omega, depth).astype(np.float64)
    cp = omega / np.maximum(k, 1e-12)
    return cp.astype(np.float32)

def kp_from_fp_tp_np(fp=None, tp=None, depth=None):
    if depth is None: raise ValueError("Need depth")
    if fp is not None:
        omega = 2*np.pi*np.clip(_as_f64_np(fp), 1e-6, None)
    else:
        omega = 2*np.pi/np.clip(_as_f64_np(tp), 1e-3, None)
    return solve_k_omega_h_np(omega, _as_f64_np(depth)).astype(np.float32)

# ---- CuPy versions (used only if --use-cupy & cupy available) ----
def solve_k_omega_h_cp(cp, omega, h, max_iter=30, tol=1e-10):
    omega = cp.asarray(omega, dtype=cp.float64)
    h = cp.asarray(h, dtype=cp.float64)
    omega = cp.clip(omega, 1e-6, None)
    h = cp.clip(h, 1e-6, None)
    k = cp.maximum(omega*omega / G, 1e-10)
    for _ in range(max_iter):
        kh = k * h
        tanh_kh = cp.tanh(kh)
        sech2_kh = 1.0 - tanh_kh * tanh_kh
        F = G * k * tanh_kh - omega*omega
        dF = G * (tanh_kh + k * h * sech2_kh)
        dF = cp.where(cp.abs(dF) < 1e-16, cp.sign(dF) * 1e-16, dF)
        step = F / dF
        k_next = k - step
        k_next = cp.clip(k_next, 1e-12, 1e6)
        if cp.all(cp.abs(step) <= tol * cp.maximum(k, 1.0)): k = k_next; break
        k = k_next
    return k.astype(cp.float32)

def cp_from_fp_tp_cp(cp_mod, fp=None, tp=None, depth=None):
    if fp is None and tp is None: raise ValueError("Need fp or tp")
    if depth is None: raise ValueError("Need depth")
    if fp is not None:
        omega = 2*np.pi*cp_mod.clip(cp_mod.asarray(fp, dtype=cp_mod.float64), 1e-6, None)
    else:
        omega = 2*np.pi/cp_mod.clip(cp_mod.asarray(tp, dtype=cp_mod.float64), 1e-3, None)
    k = solve_k_omega_h_cp(cp_mod, omega, depth)
    cpv = omega / cp_mod.maximum(k.astype(cp_mod.float64), 1e-12)
    return cpv.astype(cp_mod.float32)

def kp_from_fp_tp_cp(cp_mod, fp=None, tp=None, depth=None):
    if depth is None: raise ValueError("Need depth")
    if fp is not None:
        omega = 2*np.pi*cp_mod.clip(cp_mod.asarray(fp, dtype=cp_mod.float64), 1e-6, None)
    else:
        omega = 2*np.pi/cp_mod.clip(cp_mod.asarray(tp, dtype=cp_mod.float64), 1e-3, None)
    return solve_k_omega_h_cp(cp_mod, omega, depth).astype(cp_mod.float32)

# ------------------- xarray helpers -------------------
def open_zarr_any(path):
    try:  return xr.open_zarr(path, consolidated=True)
    except Exception: return xr.open_zarr(path, consolidated=False)

def open_mask_any(mask_store):
    da = None
    try:  da = xr.open_zarr(mask_store, consolidated=True)["ocean_mask"]
    except Exception:
        da = xr.open_zarr(mask_store, consolidated=False)["ocean_mask"]
    return da.astype(bool)

def pick(ds, names):
    for n in names:
        if n in ds: return n
    return None

def dims(ds):
    t="time"
    y="lat" if "lat" in ds.dims else ("latitude" if "latitude" in ds.dims else "y")
    x="lon" if "lon" in ds.dims else ("longitude" if "longitude" in ds.dims else "x")
    return t,y,x

# ------------------- split & sampling -------------------
def build_global_time_index(paths):
    pairs=[]; sizes={}
    for p in paths:
        ds=open_zarr_any(p)
        T=ds.sizes["time"]
        sizes[p]=T
        pairs.extend([(p,t) for t in range(T)])
    return pairs, sizes

def split_pairs_time(pairs, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng=np.random.default_rng(seed); idx=np.arange(len(pairs)); rng.shuffle(idx)
    n_train=int(train_frac*len(idx)); n_val=int(val_frac*len(idx))
    i_tr=idx[:n_train]; i_va=idx[n_train:n_train+n_val]; i_te=idx[n_train+n_val:]
    def to_by_store(idxs):
        by={}
        for i in idxs:
            path,t = pairs[int(i)]
            base=os.path.basename(path)
            by.setdefault(base, []).append(int(t))
        for k in by: by[k].sort()
        return by
    return to_by_store(i_tr), to_by_store(i_va), to_by_store(i_te)

def sample_task(task_id, seed, need, paths, train_by_store, mask_root,
                use_cupy=False, slab=32):
    """
    One parallel task: gather ~need samples (pointwise) from train timesteps.
    Returns partial sums for X and y.
    """
    # local RNG
    rng = np.random.default_rng(seed + 100000*task_id)

    # lazy open per task
    opened=[]  # (base, ds, mask, times)
    for p in paths:
        base=os.path.basename(p)
        times=train_by_store.get(base, [])
        if not times: continue
        ds=open_zarr_any(p)
        m = open_mask_any(str(Path(mask_root)/(Path(p).stem+"_mask.zarr")))
        opened.append((base, ds, m, np.array(times, dtype=np.int64)))
    if not opened:
        return dict(n=0, x_sum=np.zeros(7, np.float64), x_sq=np.zeros(7, np.float64),
                    y_sum=0.0, y_sq=0.0)

    # choose backend for cp/kp
    cp_mod = try_import_cupy(use_cupy)
    use_gpu = (cp_mod is not None)

    # accumulators
    n=0
    x_sum=np.zeros(7, np.float64); x_sq=np.zeros(7, np.float64)
    y_sum=0.0; y_sq=0.0

    while n < need:
        base, ds, m, times = opened[rng.integers(0, len(opened))]
        t,y,x = dims(ds)
        H,W = ds.sizes[y], ds.sizes[x]
        if times.size==0: continue
        ti = int(times[rng.integers(0, times.size)])
        yi = int(rng.integers(0, H-slab+1))
        xi = int(rng.integers(0, W-slab+1))

        # slice a small slab to amortize IO and vectorize physics
        sub  = ds.isel({t:slice(ti,ti+1), y:slice(yi,yi+slab), x:slice(xi,xi+slab)}).load()
        try:
            msub = m.isel({t:ti, y:slice(yi,yi+slab), x:slice(xi,xi+slab)}).load().values
        except Exception:
            msub = m.isel({y:slice(yi,yi+slab), x:slice(xi,xi+slab)}).load().values

        hsN  = pick(sub, ["hs","Hs","SWH"])
        uwN  = pick(sub, ["uwnd","uw10","u_wind","u10"])
        vwN  = pick(sub, ["vwnd","vw10","v_wind","v10"])
        u10N = pick(sub, ["U10","u10"])
        wdN  = pick(sub, ["wdir","wind_direction"])
        fpN  = pick(sub, ["fp","fpeak","peak_frequency"])
        tpN  = pick(sub, ["tp","tpeak","peak_period"])
        dN   = pick(sub, ["dpt","depth","bathymetry"])
        yN   = pick(sub, ["wcm","target"])
        if hsN is None or dN is None or (u10N is None and (uwN is None or vwN is None)) or yN is None:
            continue

        hs = sub[hsN].values.astype(np.float32)[0]     # (h,w)
        d  = sub[dN].values
        if d.ndim==3: d = d[0]
        depth = d.astype(np.float32)

        if u10N:
            U10 = sub[u10N].values.astype(np.float32)[0]
        else:
            U10 = np.sqrt(np.maximum(sub[uwN].values[0]**2 + sub[vwN].values[0]**2, 0.0)).astype(np.float32)

        if wdN:
            wdir = np.deg2rad(sub[wdN].values.astype(np.float32))[0]
            cosw, sinw = np.cos(wdir), np.sin(wdir)
        else:
            wdir = np.arctan2(sub[vwN].values[0], sub[uwN].values[0]).astype(np.float32)
            cosw, sinw = np.cos(wdir), np.sin(wdir)

        fp = sub[fpN].values.astype(np.float32)[0] if fpN else None
        tp = sub[tpN].values.astype(np.float32)[0] if tpN else None

        # compute finite-depth cp, kp (vectorized over slab)
        if (fp is None or not np.isfinite(fp).any()) and (tp is None or not np.isfinite(tp).any()):
            continue

        if use_gpu:
            # move only what’s needed
            depth_g = cp_mod.asarray(depth, dtype=cp_mod.float32)
            if fpN:
                fp_g = cp_mod.asarray(fp, dtype=cp_mod.float32)
                cpv = cp_from_fp_tp_cp(cp_mod, fp=fp_g, tp=None, depth=depth_g)
                kv  = kp_from_fp_tp_cp(cp_mod, fp=fp_g, tp=None, depth=depth_g)
            else:
                tp_g = cp_mod.asarray(tp, dtype=cp_mod.float32)
                cpv = cp_from_fp_tp_cp(cp_mod, fp=None, tp=tp_g, depth=depth_g)
                kv  = kp_from_fp_tp_cp(cp_mod, fp=None, tp=tp_g, depth=depth_g)
            cpv = cp_mod.asnumpy(cpv); kv = cp_mod.asnumpy(kv)
        else:
            cpv = cp_from_fp_tp_np(fp=fp if fpN else None, tp=tp if tpN else None, depth=depth)
            kv  = kp_from_fp_tp_np(fp=fp if fpN else None, tp=tp if tpN else None, depth=depth)

        U10c = np.clip(U10, 0.5, None)
        age  = (cpv / U10c).astype(np.float32)
        steep= (kv * hs / 2.0).astype(np.float32)
        depthv = depth.astype(np.float32)

        X = np.stack([hs, U10, cosw, sinw, age, steep, depthv], axis=-1)  # (h,w,7)
        y = sub[yN].values.astype(np.float32)[0]                           # (h,w)

        valid = np.isfinite(X).all(axis=-1) & np.isfinite(y) & (msub>0)
        if not np.any(valid): continue

        Xi = X[valid].reshape(-1,7); yi = y[valid].reshape(-1)

        # thin to keep per-task memory bounded
        need_now = min(need - n, Xi.shape[0])
        if Xi.shape[0] > need_now:
            sel = np.random.default_rng(seed + task_id).choice(Xi.shape[0], size=need_now, replace=False)
            Xi = Xi[sel]; yi = yi[sel]

        n_add = Xi.shape[0]
        n += n_add
        x_sum += Xi.sum(axis=0, dtype=np.float64)
        x_sq  += (Xi.astype(np.float64)**2).sum(axis=0)
        y_sum += float(yi.sum(dtype=np.float64))
        y_sq  += float((yi.astype(np.float64)**2).sum())

    return dict(n=int(n), x_sum=x_sum, x_sq=x_sq, y_sum=float(y_sum), y_sq=float(y_sq))

# ------------------- brace & multi-glob expansion -------------------
_BRACE_RE = re.compile(r"\{(\d+)\.\.(\d+)\}")

def _expand_brace_once(pattern):
    """
    Expand a single numeric brace range like 'ww3.{1980..2014}.zarr' into a list
    of patterns with the numbers substituted.
    """
    m = _BRACE_RE.search(pattern)
    if not m:
        return [pattern]
    a, b = int(m.group(1)), int(m.group(2))
    lo, hi = (a, b) if a <= b else (b, a)
    out = []
    for y in range(lo, hi + 1):
        out.append(pattern[:m.start()] + str(y) + pattern[m.end():])
    return out

def _expand_glob_arg(raw):
    """
    Accepts a single string that may contain one or more space-separated patterns.
    Each pattern may contain ONE numeric brace range {A..B}. We expand braces
    and then apply glob() to each expanded pattern.
    """
    patterns = raw.strip().split()
    expanded_patterns = []
    for pat in patterns:
        expanded_patterns.extend(_expand_brace_once(os.path.expanduser(pat)))
    paths = []
    for pat in expanded_patterns:
        paths.extend(sorted(glob(pat)))
    return paths

# ------------------- main -------------------
def parse():
    ap=argparse.ArgumentParser("Parallel pointwise scaler with Dask (CPU) and optional CuPy (GPU).")
    ap.add_argument("--zarr-glob", required=True,
                    help='One or more patterns (space-separated). Supports numeric brace ranges like "/dir/ww3.{1980..2014}.zarr"')
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--samples", type=int, default=500000, help="total TRAIN samples to draw")
    ap.add_argument("--target", default="wcm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)

    # parallel options
    ap.add_argument("--use-dask", action="store_true", help="enable Dask parallelism")
    ap.add_argument("--nworkers", type=int, default=8)
    ap.add_argument("--threads-per-worker", type=int, default=1)
    ap.add_argument("--scheduler", type=str, default=None, help="tcp://host:8786 to connect instead of local")
    ap.add_argument("--tasks-per-worker", type=int, default=4, help="num tasks per worker")
    ap.add_argument("--use-cupy", action="store_true", help="use CuPy for wave physics if available")
    return ap.parse_args()

def main():
    args=parse()
    # HPC friendliness for math libs
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v,"1")

    # NEW: brace & multi-glob expansion
    paths = _expand_glob_arg(args.zarr_glob)
    if not paths:
        raise SystemExit(f"No zarr stores matched: {args.zarr_glob}")

    # 1) build global time index & split
    pairs, sizes = build_global_time_index(paths)
    train_by_store, val_by_store, test_by_store = split_pairs_time(
        pairs, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )

    total = args.samples
    # 2) parallel sampling with Dask
    if args.use_dask:
        try:
            if args.scheduler:
                from dask.distributed import Client
                client = Client(args.scheduler)
                print(f"[DASK] Connected to {args.scheduler}")
            else:
                from dask.distributed import Client, LocalCluster
                cluster = LocalCluster(n_workers=args.nworkers, threads_per_worker=args.threads_per_worker,
                                       processes=True, silence_logs="WARNING")
                client = Client(cluster)
                print(f"[DASK] LocalCluster: {args.nworkers}×{args.threads_per_worker}")

            # Plan tasks
            ntasks = max(args.nworkers * args.tasks_per_worker, 1)
            per = math.ceil(total / ntasks)
            futures=[]
            for k in range(ntasks):
                fut = client.submit(
                    sample_task, k, args.seed, per, paths, train_by_store, args.mask_root,
                    use_cupy=args.use_cupy, pure=False
                )
                futures.append(fut)
            parts = client.gather(futures)
        except Exception as e:
            print(f"[WARN] Dask failed ({e}); falling back to single-process.")
            parts = [sample_task(0, args.seed, total, paths, train_by_store, args.mask_root, use_cupy=args.use_cupy)]
        finally:
            try: client.close()
            except Exception: pass
    else:
        parts = [sample_task(0, args.seed, total, paths, train_by_store, args.mask_root, use_cupy=args.use_cupy)]

    # 3) reduce partials
    n = sum(p["n"] for p in parts)
    if n == 0:
        raise SystemExit("No samples collected; check masks/variables.")
    x_sum = np.sum([p["x_sum"] for p in parts], axis=0)
    x_sq  = np.sum([p["x_sq"]  for p in parts], axis=0)
    y_sum = sum(p["y_sum"] for p in parts)
    y_sq  = sum(p["y_sq"]  for p in parts)

    mean = (x_sum / n).astype(np.float32)
    var  = np.maximum(x_sq / n - mean.astype(np.float64)**2, 0.0)
    std  = np.sqrt(var).astype(np.float32); std[std==0]=1.0

    y_mean = float(y_sum / n)
    y_var  = max(y_sq / n - y_mean**2, 0.0)
    y_std  = float(np.sqrt(y_var) if y_var>0 else 1.0)

    # 4) save JSON
    out = dict(
        mean=mean.tolist(),
        std=std.tolist(),
        y_mean=y_mean,
        y_std=y_std,
        target=args.target,
        splits=dict(
            by_store=dict(train=train_by_store, val=val_by_store, test=test_by_store),
            seed=args.seed,
            fractions=dict(train=args.train_frac, val=args.val_frac, test=args.test_frac)
        ),
        features=["hs","U10","cos(wdir)","sin(wdir)","wave_age","steepness","depth"],
        notes="Parallel scaler; finite-depth cp/kp; CuPy used: {}".format(bool(try_import_cupy(args.use_cupy)))
    )
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] n={n:,}  Wrote scaler to {args.out}")
    print("  X mean:", mean.tolist())
    print("  X std :", std.tolist())
    print(f"  y mean: {y_mean:.6g}, y std: {y_std:.6g}")

if __name__=="__main__":
    main()

