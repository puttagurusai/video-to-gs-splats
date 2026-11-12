"""Joint multi-view depth alignment (implements section 5 of Video_to_3D_Complete_Pipeline_and_Depth_Alignment.docx).

Each frame has a raw monocular map M_i (higher = nearer, "inverse-depth-like"), known camera poses and a shared K.
Per frame we calibrate  rho_i = a_i * M_i + b_i  (inverse depth),  D_i = 1 / rho_i,  in three stages:
  1. independent robust fits (RANSAC + IRLS-Huber) to sparse anchors (COLMAP landmarks: pixel + true z-depth);
  2. an overlap graph (temporal neighbours) and dense reprojection i -> j with visibility/occlusion gating;
  3. joint robust least squares over all (a_i, b_i): sparse anchors + multi-view log-depth residuals + temporal smoothness
     + prior to the independent fits. Correspondences are recomputed in an outer loop (they depend on D_i).
Conventions: T_cw maps world to camera (X_c = R X_w + t); z-depth along the camera axis (not ray length); pixel centres at +0.5.

Pure numpy/scipy (CPU). No assumption that poses or K are exact: residuals are robust and gated.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

EPS_RHO = 1e-4


def bilinear(img, u, v):
    """Sample img at continuous pixel coordinates (pixel centres at +0.5)."""
    return map_coordinates(img, [np.asarray(v) - 0.5, np.asarray(u) - 0.5], order=1, mode="nearest")


def robust_affine(m, y, iters=300, rng=None):
    """Fit y ~ a*m + b with RANSAC (2-point) then IRLS-Huber on the inliers. Returns (a, b, inlier mask) or None."""
    rng = rng or np.random.default_rng(0)
    n = len(m)
    if n < 6:
        return None
    thr = 0.04 * max(np.median(np.abs(y)), 1e-6)
    best_in, best = None, -1
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(m[i] - m[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (m[i] - m[j])
        b = y[i] - a * m[i]
        if a <= 0:
            continue
        inl = np.abs(a * m + b - y) < thr
        if inl.sum() > best:
            best, best_in = inl.sum(), inl
    if best_in is None or best_in.sum() < 5:
        return None
    A = np.stack([m, np.ones_like(m)], 1)
    w = np.ones(n) * best_in
    sol = np.array([1.0, 0.0])
    for _ in range(10):
        sol = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        r = A @ sol - y
        s = 1.4826 * np.median(np.abs(r[best_in])) + 1e-9
        w = best_in / np.maximum(1.0, np.abs(r) / (1.5 * s))
    if sol[0] <= 0:
        return None
    return float(sol[0]), float(sol[1]), best_in


def independent_fits(frames, min_inliers=12, min_span=1.3):
    """Stage 1. frames: list of dicts with M, anchors=(u, v, z). Returns theta (N,2) and a low-confidence mask."""
    N = len(frames)
    theta = np.full((N, 2), np.nan)
    for i, f in enumerate(frames):
        u, v, z = f["anchors"]
        ok = z > 0
        if ok.sum() < min_inliers:
            continue
        m, y = bilinear(f["M"], u[ok], v[ok]), 1.0 / z[ok]
        fit = robust_affine(m, y)
        if fit is None or fit[2].sum() < min_inliers:
            continue
        yi = y[fit[2]]
        if yi.max() / max(yi.min(), 1e-9) < min_span:  # not enough depth diversity: shift/scale are not separable
            continue
        theta[i] = fit[:2]
    low = np.isnan(theta[:, 0])
    if low.all():
        raise ValueError("no frame has enough anchors for an independent fit")
    idx = np.arange(N)
    for k in range(2):  # borrow from the nearest valid frames (temporal interpolation)
        theta[:, k] = np.interp(idx, idx[~low], theta[~low, k])
    return theta, low


def overlap_pairs(n, window=2, extra=()):
    """Directed pairs (i, j): temporal neighbours within +-window, plus any extra (loop-closure) pairs."""
    pairs = {(i, j) for i in range(n) for j in range(max(0, i - window), min(n, i + window + 1)) if j != i}
    for i, j in extra:
        pairs.add((i, j)); pairs.add((j, i))
    return sorted(pairs)


def build_samples(frames, theta, K, pairs, stride=4, gate=0.4, occ_tol=0.12, margin=3, keep_all=False):
    """Stage 2. Dense reprojection i -> j with the current calibration. Returns arrays for the residual and diagnostics."""
    Kinv = np.linalg.inv(K)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = {k: [] for k in ("I", "J", "Mi", "Mj", "alpha", "beta", "keep", "occ", "pu", "pv", "qu", "qv", "zj")}
    for i, j in pairs:
        Mi_img, Mj_img = frames[i]["M"], frames[j]["M"]
        H, W = Mi_img.shape
        ys, xs = np.mgrid[margin:H - margin:stride, margin:W - margin:stride]
        u, v = xs.ravel() + 0.5, ys.ravel() + 0.5
        Mi = Mi_img[ys.ravel(), xs.ravel()]
        rho = theta[i, 0] * Mi + theta[i, 1]
        valid = rho > EPS_RHO
        Di = 1.0 / np.maximum(rho, EPS_RHO)
        ray = Kinv @ np.stack([u, v, np.ones_like(u)])                                   # (3,n)
        Ri, ti, Rj, tj = frames[i]["R"], frames[i]["t"], frames[j]["R"], frames[j]["t"]
        Xw = Ri.T @ (ray * Di - ti[:, None])
        Xj = Rj @ Xw + tj[:, None]
        zj = Xj[2]
        qu, qv = fx * Xj[0] / np.maximum(zj, 1e-6) + cx, fy * Xj[1] / np.maximum(zj, 1e-6) + cy
        inside = (zj > 1e-3) & (qu > margin) & (qu < W - margin) & (qv > margin) & (qv < H - margin)
        Mj = bilinear(Mj_img, qu, qv)
        rho_j = theta[j, 0] * Mj + theta[j, 1]
        valid &= rho_j > EPS_RHO
        Dj = 1.0 / np.maximum(rho_j, EPS_RHO)
        occluded = Dj < (1.0 - occ_tol) * np.maximum(zj, 1e-6)                           # j sees something nearer: i's point is hidden
        mism = np.abs(np.log(Dj) - np.log(np.maximum(zj, 1e-6))) >= gate
        keep = inside & valid & ~occluded & ~mism
        vec = Rj @ Ri.T @ ray                                                            # z_j = alpha * D_i + beta (exact, linear in D_i)
        alpha = vec[2]
        beta = (tj - Rj @ Ri.T @ ti)[2]
        sel = np.ones_like(keep) if keep_all else keep
        n = int(sel.sum())
        out["I"].append(np.full(n, i)); out["J"].append(np.full(n, j))
        for key, arr in (("Mi", Mi), ("Mj", Mj), ("alpha", alpha), ("beta", np.full_like(alpha, beta)), ("keep", keep),
                         ("occ", occluded & inside & valid), ("pu", u), ("pv", v), ("qu", qu), ("qv", qv), ("zj", zj)):
            out[key].append(arr[sel])
    return {k: np.concatenate(v) for k, v in out.items()}


def joint_optimize(frames, K, theta0, pairs, outer=3, gates=(0.6, 0.4, 0.3), stride=4, w_sparse=1.0, w_mv=1.0,
                   lam_temp=0.3, lam_prior=0.05, max_nfev=80, held_out_pairs=(), verbose=True):
    """Stage 3. Returns theta (N,2) and a history of (outer iteration, multiview residual median, #samples)."""
    N = len(frames)
    theta = theta0.copy()
    a_scale = max(float(np.median(theta0[:, 0])), 1e-6)
    anc = []
    for i, f in enumerate(frames):
        u, v, z = f["anchors"]
        ok = z > 0
        anc.append((np.full(int(ok.sum()), i), bilinear(f["M"], u[ok], v[ok]), 1.0 / z[ok]))
    Ia, ma, ya = (np.concatenate([a[k] for a in anc]) for k in range(3))
    y_scale = max(float(np.median(ya)), 1e-6)
    hist = []
    for it in range(outer):
        S = build_samples(frames, theta, K, pairs, stride=stride, gate=gates[min(it, len(gates) - 1)])
        I, J, Mi, Mj, al, be = S["I"], S["J"], S["Mi"], S["Mj"], S["alpha"], S["beta"]
        n_mv, n_sp = len(I), len(Ia)

        def resid(x):
            th = x.reshape(N, 2)
            a, b = th[:, 0], th[:, 1]
            Di = 1.0 / np.maximum(a[I] * Mi + b[I], EPS_RHO)
            zj = np.maximum(al * Di + be, 1e-6)
            Dj = 1.0 / np.maximum(a[J] * Mj + b[J], EPS_RHO)
            r_mv = w_mv * (np.log(Dj) - np.log(zj)) / np.sqrt(max(n_mv, 1) / 1000.0)
            r_sp = w_sparse * (a[Ia] * ma + b[Ia] - ya) / y_scale / np.sqrt(max(n_sp, 1) / 1000.0)
            r_t = lam_temp * np.concatenate([(a[1:] - a[:-1]) / a_scale, (b[1:] - b[:-1]) / y_scale])
            r_p = lam_prior * np.concatenate([(a - theta0[:, 0]) / a_scale, (b - theta0[:, 1]) / y_scale])
            return np.concatenate([r_mv, r_sp, r_t, r_p])

        rows, cols = [], []
        rr = np.arange(n_mv)
        for who in (I, J):          # each multi-view residual depends on (a, b) of frames i and j
            for c in (0, 1):
                rows.append(rr); cols.append(2 * who + c)
        r0 = n_mv
        rr = r0 + np.arange(n_sp)
        for c in (0, 1):
            rows.append(rr); cols.append(2 * Ia + c)
        r0 += n_sp
        for k in range(N - 1):
            for c in (0, 1):
                for who in (k, k + 1):
                    rows.append(np.array([r0 + c * (N - 1) + k])); cols.append(np.array([2 * who + c]))
        r0 += 2 * (N - 1)
        for k in range(N):
            for c in (0, 1):
                rows.append(np.array([r0 + c * N + k])); cols.append(np.array([2 * k + c]))
        total = r0 + 2 * N
        sparsity = coo_matrix((np.ones(sum(len(r) for r in rows), dtype=np.int8), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(total, 2 * N)).tocsr()
        before = float(np.median(np.abs(resid(theta.ravel())[:n_mv]))) if n_mv else float("nan")
        sol = least_squares(resid, theta.ravel(), jac_sparsity=sparsity, loss="soft_l1", f_scale=0.1, x_scale="jac",
                            max_nfev=max_nfev)
        theta = sol.x.reshape(N, 2)
        after = float(np.median(np.abs(resid(theta.ravel())[:n_mv]))) if n_mv else float("nan")
        hist.append((it, before, after, n_mv))
        if verbose:
            print(f"  outer {it}: {n_mv} multi-view samples, median |residual| {before:.4f} -> {after:.4f}")
    return theta, hist


def apply_calibration(frames, theta):
    """Refined z-depth maps D_i = 1 / max(a_i M_i + b_i, eps) and the share of clamped (non-positive rho) pixels."""
    depths, clamped = [], []
    for i, f in enumerate(frames):
        rho = theta[i, 0] * f["M"] + theta[i, 1]
        clamped.append(float((rho <= EPS_RHO).mean()))
        depths.append(1.0 / np.maximum(rho, EPS_RHO))
    return depths, clamped


def heldout_residual(frames, K, theta, pairs, stride=4, gate=0.6):
    """|log D_j(q) - log z_j| over gated samples of pairs NOT used for optimisation (generalisation check).
    Returns (median, P90, n). The spec asks for both: the median is dominated by well-fitted frames, P90 shows the tail."""
    S = build_samples(frames, theta, K, pairs, stride=stride, gate=gate)
    if len(S["I"]) == 0:
        return float("nan"), float("nan"), 0
    Di = 1.0 / np.maximum(theta[S["I"], 0] * S["Mi"] + theta[S["I"], 1], EPS_RHO)
    zj = np.maximum(S["alpha"] * Di + S["beta"], 1e-6)
    Dj = 1.0 / np.maximum(theta[S["J"], 0] * S["Mj"] + theta[S["J"], 1], EPS_RHO)
    r = np.abs(np.log(Dj) - np.log(zj))
    return float(np.median(r)), float(np.percentile(r, 90)), len(S["I"])
