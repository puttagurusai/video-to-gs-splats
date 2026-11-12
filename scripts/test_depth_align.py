"""Synthetic unit test for depth_align.py (spec section 14): known geometry, known poses, corrupted per-frame inverse depth.

  python -I scripts/test_depth_align.py

Scene: floor + back wall + side wall + an occluding box, 14 cameras translating sideways with small yaw changes.
Corruption per frame: random scale a_true in [0.65, 1.6], shift b_true, smooth low-frequency warp, high-frequency noise.
Anchors: ~120 sparse points per frame (1% depth noise, 10% outliers); a few frames get only 7 anchors (too weak to fit alone).
Checks: recovery of the shared surface (esp. weak-anchor frames), positive depth, occlusion rejection, held-out residual.
"""
import sys
import os

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_align as da

W, H, F = 192, 144, 200.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
rng = np.random.default_rng(7)


def cast_depth(R, t):
    """Analytic z-depth of the synthetic scene for a camera (X_c = R X_w + t). Returns (H,W) float, inf where nothing is hit."""
    C = -R.T @ t
    ys, xs = np.mgrid[0:H, 0:W]
    ray_c = np.stack([(xs + 0.5 - K[0, 2]) / F, (ys + 0.5 - K[1, 2]) / F, np.ones_like(xs, dtype=float)], -1)
    d = ray_c @ R                                                                  # world directions (z-component in camera = 1 -> t = z-depth)
    best = np.full((H, W), np.inf)

    def plane(axis, val, sign):
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (val - C[axis]) / d[..., axis]
        ok = (tt > 0.05) & (np.sign(d[..., axis]) == sign)
        return np.where(ok, tt, np.inf)

    for tt in (plane(1, 1.5, 1), plane(2, 7.0, 1), plane(0, 3.2, 1), plane(0, -3.2, -1)):
        best = np.minimum(best, tt)
    lo, hi = np.array([-0.5, 0.3, 3.0]), np.array([0.7, 1.5, 4.2])                 # occluding box (sits on the floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - C) / d, (hi - C) / d
    tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
    hit = (tf > np.maximum(tn, 0.05)) & (tn > 0.05)
    best = np.minimum(best, np.where(hit, tn, np.inf))
    return best


def make_scene(n=14):
    frames, gts = [], []
    for i in range(n):
        x = -1.2 + 2.4 * i / (n - 1)
        yaw = np.radians(-4 + 8 * i / (n - 1))
        Rc2w = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R = Rc2w.T
        t = -R @ np.array([x, 0.0, 0.0])
        z = cast_depth(R, t)
        gts.append(z)
        rho = 1.0 / np.where(np.isfinite(z), z, 50.0)
        a_true, b_true = rng.uniform(0.65, 1.6), rng.uniform(-0.04, 0.04)
        low = gaussian_filter(rng.normal(size=(H, W)), 14) * 0.2 * np.median(rho)
        low /= max(np.abs(low).max(), 1e-9) / (0.03 * np.median(rho))
        M = (rho - b_true) / a_true + low + rng.normal(size=(H, W)) * 0.004 * np.median(rho)
        n_anc = 7 if i in (3, 9) else 120
        ua, va = rng.uniform(4, W - 4, n_anc), rng.uniform(4, H - 4, n_anc)
        za = da.bilinear(np.where(np.isfinite(z), z, 50.0), ua, va) * (1 + rng.normal(size=n_anc) * 0.01)
        bad = rng.random(n_anc) < 0.10
        za[bad] *= rng.uniform(0.5, 2.0, bad.sum())
        frames.append(dict(M=M, R=R, t=t, anchors=(ua, va, za)))
    return frames, gts


def depth_error(frames, theta, gts, idx):
    depths, _ = da.apply_calibration(frames, theta)
    errs = []
    for i in idx:
        ok = np.isfinite(gts[i]) & (gts[i] < 20)
        errs.append(np.median(np.abs(depths[i][ok] - gts[i][ok]) / gts[i][ok]))
    return float(np.mean(errs))


def main():
    frames, gts = make_scene()
    n = len(frames)
    weak = [3, 9]
    theta0, low = da.independent_fits(frames)
    print(f"stage 1: independent fits; low-confidence (borrowed) frames: {np.nonzero(low)[0].tolist()} (weak-anchor frames: {weak})")
    opt_pairs = da.overlap_pairs(n, window=2)
    held_pairs = [p for p in da.overlap_pairs(n, window=5) if p not in set(opt_pairs)]
    theta, hist = da.joint_optimize(frames, K, theta0, opt_pairs)

    all_idx = list(range(n))
    e0, e1 = depth_error(frames, theta0, gts, all_idx), depth_error(frames, theta, gts, all_idx)
    w0, w1 = depth_error(frames, theta0, gts, weak), depth_error(frames, theta, gts, weak)
    h0, h0p, hn0 = da.heldout_residual(frames, K, theta0, held_pairs)
    h1, h1p, hn1 = da.heldout_residual(frames, K, theta, held_pairs)
    depths, clamped = da.apply_calibration(frames, theta)

    # occlusion rejection vs ground truth
    S = da.build_samples(frames, theta, K, opt_pairs, gate=1e9, keep_all=True)
    occ_true, occ_pred, tot = 0, 0, 0
    for (i, j) in opt_pairs:
        m = (S["I"] == i) & (S["J"] == j)
        if not m.any():
            continue
        zi_gt = da.bilinear(np.where(np.isfinite(gts[i]), gts[i], 50.0), S["pu"][m], S["pv"][m])
        gt_j = da.bilinear(np.where(np.isfinite(gts[j]), gts[j], 50.0), S["qu"][m], S["qv"][m])
        Rj, tj, Ri, ti = frames[j]["R"], frames[j]["t"], frames[i]["R"], frames[i]["t"]
        ray = np.linalg.inv(K) @ np.stack([S["pu"][m], S["pv"][m], np.ones(m.sum())])
        z_true_in_j = (Rj @ (Ri.T @ (ray * zi_gt - ti[:, None])) + tj[:, None])[2]
        truly_occluded = (gt_j < 0.9 * z_true_in_j) & (zi_gt < 20)
        occ_true += int(truly_occluded.sum())
        occ_pred += int((truly_occluded & S["occ"][m]).sum())
        tot += int(m.sum())
    occ_recall = occ_pred / max(occ_true, 1)

    print(f"\nmedian relative depth error, all frames : {100 * e0:.2f}%  ->  {100 * e1:.2f}%")
    print(f"median relative depth error, weak frames: {100 * w0:.2f}%  ->  {100 * w1:.2f}%   (frames {weak}, 7 anchors each)")
    print(f"held-out |log depth ratio| (pairs not optimised, {hn0}/{hn1} samples): median {h0:.4f} -> {h1:.4f}; P90 {h0p:.4f} -> {h1p:.4f}")
    print(f"clamped (non-positive rho) pixels: max {100 * max(clamped):.2f}% ; min rho positive after clamp: {all(d.min() > 0 for d in depths)}")
    print(f"occlusion rejection: {occ_pred}/{occ_true} truly-occluded samples were rejected ({100 * occ_recall:.0f}%) of {tot} samples")

    checks = [
        ("overall depth error does not get worse and is < 6%", e1 <= e0 * 1.05 and e1 < 0.06),
        ("weak-anchor frames improve by >= 25%", w1 <= 0.75 * w0),
        ("held-out residual: median not worse (<= +5%)", h1 <= 1.05 * h0),
        ("held-out residual: P90 drops by >= 20% (spec asks for median and P90)", h1p <= 0.8 * h0p),
        ("all depths strictly positive", all(d.min() > 0 for d in depths)),
        ("clamped pixels < 2%", max(clamped) < 0.02),
        ("occlusion rejection recall >= 60%", occ_recall >= 0.6),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok_all = all(c[1] for c in checks)
    print("\nRESULT:", "ALL PASS" if ok_all else "FAILURES")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
