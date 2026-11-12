"""Compare renders (scripts/render_views.py) with the real held-out video frames: coverage, SSIM, PSNR.

  python -I scripts/eval_views.py --views export/room_relief/views.json --prefix export/room_eval_relief

Metrics are computed only where the render has geometry; `coverage` says how much of the frame that is.
`match` = coverage x SSIM (a single number: how much of the real frame is reproduced and how faithfully).
These are image-similarity measures, not a human realism rating; use them to compare methods on the same views.
Also writes <prefix>_cmp.png (real | render, up to 3 views stacked).
"""
import argparse
import json
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    views = json.load(open(a.views))
    rows, tiles = [], []
    for v in views:
        p = f"{a.prefix}_{os.path.splitext(v['name'])[0]}.png"
        if not os.path.isfile(p):
            print("missing render:", p)
            continue
        ren = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        real = cv2.imread(v["frame"])
        real = cv2.resize(real, (ren.shape[1], ren.shape[0]), interpolation=cv2.INTER_AREA)
        mask = ren[..., 3] > 127
        cov = float(mask.mean())
        if mask.sum() < 100:
            rows.append((v["name"], cov, float("nan"), float("nan"), float("nan")))
            continue
        rgb = ren[..., :3]
        mse = float(np.mean((real[mask].astype(np.float64) - rgb[mask].astype(np.float64)) ** 2))
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
        _, smap = structural_similarity(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY),
                                        data_range=255, full=True)
        gr, gp = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, bmap = structural_similarity(cv2.GaussianBlur(gr, (0, 0), 3), cv2.GaussianBlur(gp, (0, 0), 3), data_range=255, full=True)
        rows.append((v["name"], cov, float(smap[mask].mean()), float(psnr), float(bmap[mask].mean())))
        on_grey = np.where(mask[..., None], rgb, np.full_like(rgb, 128))
        tiles.append(np.hstack([real, on_grey]))
    print(f"{'frame':18s} {'coverage':>8s} {'SSIM':>6s} {'PSNR':>6s} {'blurSSIM':>9s}")
    for n, c, s, p, b in rows:
        print(f"{n:18s} {c:8.2f} {s:6.3f} {p:6.1f} {b:9.3f}")
    ok = [r for r in rows if not np.isnan(r[2])]
    if ok:
        cov, ssim, psnr, bssim = (float(np.mean([r[i] for r in ok])) for i in (1, 2, 3, 4))
        print(f"{'MEAN':18s} {cov:8.2f} {ssim:6.3f} {psnr:6.1f} {bssim:9.3f}   match (coverage x SSIM) = {cov * ssim:.3f}; "
              f"coarse match (coverage x blurSSIM) = {cov * bssim:.3f}")
    if tiles:
        h = min(t.shape[0] for t in tiles)
        cv2.imwrite(a.prefix + "_cmp.png", np.vstack([cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles[:3]]))


if __name__ == "__main__":
    main()
