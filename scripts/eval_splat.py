"""Score Brush's held-out renders (--eval-save-to-disk) against the real frames with the SAME metrics as eval_views.py.

  python -I scripts/eval_splat.py --renders export/brush_run1/eval_6000 --frames colmap_mine/dense/images

Both images are resized to the size eval_views.py uses (0.5 x the real frame), then: SSIM (grayscale), blurred SSIM
(Gaussian sigma 3, coarse structure) and PSNR over the whole image (a splat render has no empty background, coverage = 1).
Brush's own PSNR/SSIM use a different implementation and resolution, so use this table to compare with the mesh path.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.renders, "*.png"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        cand = [os.path.join(a.frames, stem + e) for e in (".jpg", ".png", ".jpeg")]
        real_p = next((c for c in cand if os.path.isfile(c)), None)
        if real_p is None:
            continue
        real = cv2.imread(real_p)
        size = (int(round(real.shape[1] * a.scale)), int(round(real.shape[0] * a.scale)))
        real = cv2.resize(real, size, interpolation=cv2.INTER_AREA)
        ren = cv2.imread(p, cv2.IMREAD_COLOR)
        ren = cv2.resize(ren, size, interpolation=cv2.INTER_AREA)
        mse = float(np.mean((real.astype(np.float64) - ren.astype(np.float64)) ** 2))
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
        gr, gp = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY)
        ssim = structural_similarity(gr, gp, data_range=255)
        bssim = structural_similarity(cv2.GaussianBlur(gr, (0, 0), 3), cv2.GaussianBlur(gp, (0, 0), 3), data_range=255)
        rows.append((stem, 1.0, ssim, psnr, bssim))
    print(f"{'frame':18s} {'coverage':>8s} {'SSIM':>6s} {'PSNR':>6s} {'blurSSIM':>9s}")
    for n, c, s, p, b in rows:
        print(f"{n:18s} {c:8.2f} {s:6.3f} {p:6.1f} {b:9.3f}")
    if rows:
        m = [float(np.mean([r[i] for r in rows])) for i in (1, 2, 3, 4)]
        print(f"{'MEAN':18s} {m[0]:8.2f} {m[1]:6.3f} {m[2]:6.1f} {m[3]:9.3f}   ({len(rows)} frames)")


if __name__ == "__main__":
    main()
