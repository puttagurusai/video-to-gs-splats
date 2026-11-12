"""Apply the joint multi-view depth alignment (scripts/depth_align.py) to a real COLMAP reconstruction.

Anchors = COLMAP landmarks (pixel + true z-depth in the camera). Raw maps M_i = Depth-Anything-V2 relative depth
(higher = nearer, inverse-depth-like). Poses/intrinsics come from the undistorted COLMAP model (PINHOLE).

  python -I scripts/align_real.py --work colmap_mine/dense --out export/mine_aligned.npz

Prints: flip check (a depth-flipped SfM model is rejected), anchor fit before/after, held-out cross-view residual
(median and P90) before/after, and the share of clamped pixels. Writes a bundle compatible with the texturing scripts
(names, R, t, K, hw, depth, frames_dir); depth is in COLMAP units (arbitrary global scale) unless a reference is supplied.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_align as da


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="undistorted COLMAP folder (images/, sparse/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", type=int, default=448, help="working resolution (longer side, px)")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--exclude-every", type=int, default=0,
                    help="hold out every Nth frame (by frame number) from everything downstream, for fair evaluation")
    a = ap.parse_args()
    if not a.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import pycolmap
    import torch
    from PIL import Image
    from transformers import pipeline

    rec = None
    for cand in (os.path.join(a.work, "sparse"), os.path.join(a.work, "sparse", "0")):
        if os.path.isfile(os.path.join(cand, "images.bin")):
            rec = pycolmap.Reconstruction(cand)
            break
    if rec is None:
        raise SystemExit(f"no COLMAP model under {a.work}/sparse")
    items = sorted(rec.images.items(), key=lambda kv: kv[1].name)
    if a.exclude_every:  # true held-out protocol: these frames never enter alignment, refinement, fusion or texturing
        import re
        items = [kv for kv in items if int(re.findall(r"\d+", kv[1].name)[-1]) % a.exclude_every != 0]
    print(f"{len(items)} images used ({'excluding every ' + str(a.exclude_every) + 'th frame' if a.exclude_every else 'all registered'}), "
          f"{len(rec.points3D)} points")

    pipe = pipeline("depth-estimation", model=a.model, device=0 if torch.cuda.is_available() else -1)
    img_dir = os.path.join(a.work, "images")
    frames, names, Ks, flips = [], [], [], []
    for _, im in items:
        cam = rec.cameras[im.camera_id]
        p = np.asarray(cam.params, dtype=np.float64)
        fx, fy, cx, cy = (p[0], p[1], p[2], p[3]) if len(p) == 4 else (p[0], p[0], p[1], p[2])
        bgr = cv2.imread(os.path.join(img_dir, im.name))
        h0, w0 = bgr.shape[:2]
        s = min(1.0, a.side / max(h0, w0))
        W, H = int(round(w0 * s)), int(round(h0 * s))
        sx, sy = W / cam.width, H / cam.height
        out = pipe(Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))["predicted_depth"]
        M = out.squeeze().float().cpu().numpy() if torch.is_tensor(out) else np.asarray(out, dtype=np.float32).squeeze()
        M = cv2.resize(M, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Mx = np.asarray(pose.matrix() if callable(pose.matrix) else pose.matrix, dtype=np.float64)
        R, t = Mx[:, :3], Mx[:, 3]
        us, vs, zs = [], [], []
        for p2 in im.points2D:
            ok = p2.has_point3D() if callable(p2.has_point3D) else p2.has_point3D
            if not ok:
                continue
            pt = rec.points3D[p2.point3D_id]
            if pt.track.length() < a.min_track:
                continue
            z = float((R @ pt.xyz + t)[2])
            if z > 0:
                us.append(p2.xy[0] * sx); vs.append(p2.xy[1] * sy); zs.append(z)
        us, vs, zs = map(np.asarray, (us, vs, zs))
        if len(zs) >= 10:
            flips.append(spearmanr(zs, da.bilinear(M, us, vs))[0])
        frames.append(dict(M=M, R=R, t=t, anchors=(us, vs, zs)))
        names.append(im.name)
        Ks.append(np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]]))
    del pipe
    K = np.median(np.stack(Ks), axis=0)
    H, W = frames[0]["M"].shape
    n = len(frames)
    print(f"working res {W}x{H}; K fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}; anchors/frame median {int(np.median([len(f['anchors'][2]) for f in frames]))}")

    fl = float(np.median(flips)) if flips else float("nan")
    print(f"flip check: median Spearman(sparse z, relative depth M) = {fl:+.2f}  "
          f"(strongly negative = correct near/far order; positive = DEPTH-FLIPPED model)")
    if not fl < -0.3:
        raise SystemExit("REJECTED: SfM model is depth-flipped or uninformative; do not use it.")

    theta0, low = da.independent_fits(frames)
    pairs = da.overlap_pairs(n, window=a.window)
    held = [p for p in da.overlap_pairs(n, window=a.window + 3) if p not in set(pairs)]

    def anchor_err(theta):
        errs = []
        for i, f in enumerate(frames):
            u, v, z = f["anchors"]
            if len(z) < 5:
                continue
            d = 1.0 / np.maximum(theta[i, 0] * da.bilinear(f["M"], u, v) + theta[i, 1], da.EPS_RHO)
            errs.append(np.median(np.abs(d - z) / z))
        return float(np.median(errs))

    print(f"stage 1 done; low-confidence frames (borrowed fit): {int(low.sum())}/{n}")
    theta, hist = da.joint_optimize(frames, K, theta0, pairs)
    h0 = da.heldout_residual(frames, K, theta0, held)
    h1 = da.heldout_residual(frames, K, theta, held)
    depths, clamped = da.apply_calibration(frames, theta)
    print(f"\nanchor depth error (median relative): {100 * anchor_err(theta0):.1f}% -> {100 * anchor_err(theta):.1f}%")
    print(f"held-out cross-view |log depth ratio| ({h0[2]}/{h1[2]} samples): median {h0[0]:.4f} -> {h1[0]:.4f}; P90 {h0[1]:.4f} -> {h1[1]:.4f}")
    print(f"clamped (non-positive rho) pixels: max {100 * max(clamped):.2f}%; frames: {n}")

    np.savez_compressed(a.out, names=np.array(names), R=np.stack([f["R"] for f in frames]), t=np.stack([f["t"] for f in frames]),
                        K=K, hw=np.array([H, W]), depth=np.stack(depths).astype(np.float16), theta=theta,
                        frames_dir=os.path.abspath(img_dir))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
