"""Quick sanity gate for a COLMAP model: is the near/far order correct (not depth-flipped)?

A global-SfM model can reproject well (low pixel error) and still be depth-flipped. Compare sparse landmark depth with
Depth-Anything-V2 relative depth (higher = nearer): their Spearman correlation must be clearly NEGATIVE.

  python -I scripts/sfm_check.py --work colmap/dense [--samples 12] [--threshold -0.3]

Prints one JSON line; exit code 0 = pass, 2 = fail (flipped or uninformative), 1 = error.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="undistorted COLMAP folder (images/, sparse/)")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=-0.3)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--min-track", type=int, default=3)
    a = ap.parse_args()
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
        print(json.dumps(dict(error="no COLMAP model found")))
        return 1
    items = sorted(rec.images.items(), key=lambda kv: kv[1].name)
    pick = [items[i] for i in np.unique(np.linspace(0, len(items) - 1, min(a.samples, len(items))).round().astype(int))]
    pipe = pipeline("depth-estimation", model=a.model, device=0 if torch.cuda.is_available() else -1)
    vals = []
    for _, im in pick:
        cam = rec.cameras[im.camera_id]
        bgr = cv2.imread(os.path.join(a.work, "images", im.name))
        h0, w0 = bgr.shape[:2]
        s = min(1.0, 448 / max(h0, w0))
        W, H = int(round(w0 * s)), int(round(h0 * s))
        sx, sy = W / cam.width, H / cam.height
        out = pipe(Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))["predicted_depth"]
        M = out.squeeze().float().cpu().numpy() if torch.is_tensor(out) else np.asarray(out, dtype=np.float32).squeeze()
        M = cv2.resize(M, (W, H), interpolation=cv2.INTER_LINEAR)
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
        if len(zs) >= 10:
            xi = np.clip(np.round(np.asarray(us)).astype(int), 0, W - 1)
            yi = np.clip(np.round(np.asarray(vs)).astype(int), 0, H - 1)
            vals.append(float(spearmanr(np.asarray(zs), M[yi, xi])[0]))
    if not vals:
        print(json.dumps(dict(error="too few anchors to check")))
        return 2
    med = float(np.median(vals))
    ok = med < a.threshold
    print(json.dumps(dict(median_spearman=round(med, 3), frames_checked=len(vals), threshold=a.threshold, passed=ok)))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
