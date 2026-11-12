"""Write a views.json (camera poses in a mesh's final frame) for chosen frames, e.g. the held-out ones.

  python -I scripts/views_for_frames.py --poses export/mine_aligned.npz --mesh-bundle export/mine_fx_cams.npz --every 10 --out export/mine_fx_views.json

--poses       bundle that contains the poses of ALL frames (names, R, t, K, hw, frames_dir)
--mesh-bundle bundle written together with the mesh (provides world_to_mesh, i.e. the mesh's final frame)
Frames are chosen by frame number (every Nth, matching Brush's --eval-split-every N and align_real --exclude-every N).
Output format = relief_scene.py views (consumed by render_views.py / eval_views.py).
"""
import argparse
import json
import math
import os
import re

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--mesh-bundle", required=True)
    ap.add_argument("--every", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P, Mb = np.load(a.poses), np.load(a.mesh_bundle)
    names = [str(x) for x in P["names"]]
    R, T = P["R"].astype(np.float64), P["t"].astype(np.float64)
    K = P["K"].astype(np.float64)
    H, W = [int(x) for x in P["hw"]]
    fdir = str(P["frames_dir"])
    M = Mb["world_to_mesh"].astype(np.float64)
    M3, mt = M[:3, :3], M[:, 3][:3]
    s = float(np.cbrt(abs(np.linalg.det(M3))))
    Rot = M3 / s
    views = []
    for k, nm in enumerate(names):
        if int(re.findall(r"\d+", nm)[-1]) % a.every != 0:
            continue
        C = (-R[k].T @ T[k]) @ M3.T + mt
        Rb = (Rot @ R[k].T) @ np.diag([1.0, -1.0, -1.0])
        im = cv2.imread(os.path.join(fdir, nm))
        views.append(dict(name=nm, frame=os.path.abspath(os.path.join(fdir, nm)), location=C.tolist(), rotation=Rb.tolist(),
                          angle=2 * math.atan(max(W, H) / (2 * K[0, 0])), W=int(im.shape[1]), H=int(im.shape[0])))
    with open(a.out, "w") as f:
        json.dump(views, f, indent=1)
    print(f"wrote {len(views)} views to {a.out}")


if __name__ == "__main__":
    main()
