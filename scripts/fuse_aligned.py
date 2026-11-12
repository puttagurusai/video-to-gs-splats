"""Fuse the aligned depth from scripts/align_real.py into a mesh (shared GPU TSDF code) and save the camera bundle.

  python -I scripts/fuse_aligned.py --aligned export/mine_aligned.npz --out export/mine_c.ply

Scale: COLMAP units are arbitrary, so the scene is scaled so that the median depth is 1.0 ("scene units") unless
--scale-mult / --ceiling-height give a real reference. Writes <out> (PLY), <out stem>.json, <out stem>_cams.npz
(names, R, t, K, hw, filtered depth, world_to_mesh, frames_dir) for relief_scene.py / texture_blend.py.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-mult", type=float, default=1.0, help="multiplier on the unit-median-depth scale")
    ap.add_argument("--ceiling-height", type=float, default=0.0)
    ap.add_argument("--voxel", default="auto")
    ap.add_argument("--trunc-voxels", type=float, default=4.0)
    ap.add_argument("--max-voxels", type=int, default=30_000_000)
    ap.add_argument("--min-weight", type=float, default=0.2)
    ap.add_argument("--min-component-frac", type=float, default=0.02)
    ap.add_argument("--edge-grad", type=float, default=0.3)
    ap.add_argument("--max-depth-rel", type=float, default=4.0, help="drop depth beyond this many median depths")
    ap.add_argument("--consist-k", type=int, default=3)
    ap.add_argument("--consist-tol", type=float, default=0.08)
    ap.add_argument("--consist-min", type=int, default=2)
    ap.add_argument("--min-frame-agree", type=float, default=0.3)
    a = ap.parse_args()

    import torch
    from depth_odometry import consistency_filter
    from fuse import fuse_mesh_export

    B = np.load(a.aligned)
    names = [str(x) for x in B["names"]]
    R, T = B["R"].astype(np.float64), B["t"].astype(np.float64)
    K = B["K"].astype(np.float64)
    H, W = [int(x) for x in B["hw"]]
    depth = B["depth"].astype(np.float32)
    fdir = str(B["frames_dir"])
    n = len(names)
    med = float(np.median(depth[depth > 0][::97]))
    scale = a.scale_mult / med
    print(f"{n} frames {W}x{H}; median depth {med:.3f} COLMAP units -> scene scale x{scale:.4f} (median depth = {a.scale_mult:g})", flush=True)

    good = []
    for k in range(n):
        d = depth[k].copy()
        gx, gy = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        d[(np.hypot(gx, gy) / np.maximum(d, 1e-6)) > a.edge_grad] = 0
        d[(d <= 0) | (d > a.max_depth_rel * med)] = 0
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(fdir, names[k])), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        good.append(dict(R=R[k], t=T[k], K=K, W=W, H=H, rgb=rgb, depth=d, name=names[k]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kept = consistency_filter(good, a.consist_k, a.consist_tol, a.consist_min, dev)
    bad = [g["name"] for g in good if g["kept_frac"] < a.min_frame_agree]
    if bad and n - len(bad) >= 6:
        print(f"dropping {len(bad)} frame(s) that disagree with their neighbours: {bad[:6]}", flush=True)
        good = [g for g in good if g["kept_frac"] >= a.min_frame_agree]
    pts = []
    for g in good:
        sub = g["depth"][::16, ::16]
        yy, xx = np.nonzero(sub > 0)
        z = sub[yy, xx].astype(np.float64)
        Xc = np.stack([(xx * 16.0 + 0.5 - K[0, 2]) / K[0, 0] * z, (yy * 16.0 + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        pts.append((Xc - g["t"]) @ g["R"])
    pts = np.concatenate(pts)

    mesh = fuse_mesh_export(good, pts, scale, a, dict(method="colmap+aligned", frames=len(good), scale_units=scale,
                                                      kept_filter=kept))
    stem = os.path.splitext(a.out)[0]
    np.savez_compressed(stem + "_cams.npz", names=np.array([g["name"] for g in good]), R=np.stack([g["R"] for g in good]),
                        t=np.stack([g["t"] for g in good]), K=K, hw=np.array([H, W]),
                        depth=np.stack([g["depth"] for g in good]).astype(np.float16),
                        world_to_mesh=mesh.metadata["world_to_mesh"], frames_dir=fdir)
    print(f"wrote camera bundle {stem}_cams.npz", flush=True)


if __name__ == "__main__":
    main()
