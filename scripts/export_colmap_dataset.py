"""Export a DA3 camera bundle as a COLMAP-format dataset (images/ + sparse/0) that Brush / other splat trainers can read.

Use when COLMAP cannot register a clip but DA3 gives poses. Cameras are PINHOLE with SQUARE pixels: the focal length is
chosen photometrically (photo_calib.py) or given as a fraction of the longer image side. The initial point cloud is
sampled from the DA3 depth (colours from the frames).

  python -I scripts/export_colmap_dataset.py --cams export/room_t_cams.npz --out datasets/hotel [--focal auto] [--points 150000]

Writes <out>/images/*.jpg (copies of the frames used), <out>/sparse_txt/ (text model) and, if a COLMAP executable is
found, <out>/sparse/0/ (binary model made with `colmap model_converter`; otherwise the text model is placed there).
Poses/units are DA3's (arbitrary global scale). No claim that DA3 poses are as accurate as a COLMAP bundle adjustment.
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--focal", default="auto", help="'auto' (photometric scan), 'da3' (DA3 fx/fy), or a fraction of the longer side")
    ap.add_argument("--points", type=int, default=150000)
    ap.add_argument("--colmap-exe", default="")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} already exists; not overwriting.")

    B = np.load(a.cams)
    names = [str(x) for x in B["names"]]
    R, T = B["R"].astype(np.float64), B["t"].astype(np.float64)
    Kp = B["K"].astype(np.float64)
    Hp, Wp = [int(x) for x in B["hw"]]
    depth = B["depth"].astype(np.float32)
    fdir = a.frames or str(B["frames_dir"])
    n = len(names)
    first = cv2.imread(os.path.join(fdir, names[0]))
    Ho, Wo = first.shape[:2]
    print(f"{n} frames; DA3 working res {Wp}x{Hp}; images {Wo}x{Ho}; DA3 fx={Kp[0, 0]:.1f} fy={Kp[1, 1]:.1f}", flush=True)

    if a.focal == "da3":
        fxp, fyp = Kp[0, 0], Kp[1, 1]
    else:
        if a.focal == "auto":
            import torch
            import photo_calib
            grays = np.stack([cv2.resize(cv2.imread(os.path.join(fdir, nm), cv2.IMREAD_GRAYSCALE), (Wp, Hp), interpolation=cv2.INTER_AREA)
                              .astype(np.float32) / 255.0 for nm in names])
            f_iso, table = photo_calib.calibrate_isotropic_focal(depth, R, T, Kp[0, 2], Kp[1, 2], grays,
                                                                 "cuda" if torch.cuda.is_available() else "cpu")
            if f_iso is None:
                f_iso = float(np.sqrt(Kp[0, 0] * Kp[1, 1]))
                print("focal scan had too few usable pairs; using the geometric mean of DA3's fx, fy")
            else:
                print(f"photometric focal scan: best f = {f_iso:.1f} px at {Wp}x{Hp} ({f_iso / max(Wp, Hp):.3f} x longer side)")
        else:
            f_iso = float(a.focal) * max(Wp, Hp)
        fxp = fyp = f_iso
    # square pixels in the ORIGINAL image: one focal length from the longer side
    f_orig = float(np.sqrt(fxp * fyp)) * (max(Wo, Ho) / max(Wp, Hp))
    cx, cy = Wo / 2.0, Ho / 2.0
    print(f"exported camera: PINHOLE {Wo}x{Ho}, f = {f_orig:.1f} px ({f_orig / max(Wo, Ho):.3f} x longer side), principal point at the centre")

    img_dir = os.path.join(a.out, "images")
    txt_dir = os.path.join(a.out, "sparse_txt")
    os.makedirs(img_dir)
    os.makedirs(txt_dir)
    for nm in names:
        shutil.copy(os.path.join(fdir, nm), os.path.join(img_dir, nm))

    # points: sample depth pixels, back-project with the exported focal, colour from the frame
    Kx = np.array([[fxp, 0, Kp[0, 2]], [0, fyp, Kp[1, 2]], [0, 0, 1.0]])
    per = max(50, a.points // n)
    rng = np.random.default_rng(0)
    pts, cols = [], []
    for k in range(n):
        ys, xs = np.nonzero(depth[k] > 0)
        if len(ys) == 0:
            continue
        sel = rng.choice(len(ys), min(per, len(ys)), replace=False)
        ys, xs = ys[sel], xs[sel]
        z = depth[k][ys, xs].astype(np.float64)
        Xc = np.stack([(xs + 0.5 - Kx[0, 2]) / Kx[0, 0] * z, (ys + 0.5 - Kx[1, 2]) / Kx[1, 1] * z, z], 1)
        pts.append((Xc - T[k]) @ R[k])
        im = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(fdir, names[k])), (Wp, Hp), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        cols.append(im[ys, xs])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    # drop far outliers (beyond the 99th percentile distance from the median)
    d = np.linalg.norm(pts - np.median(pts, 0), axis=1)
    keep = d < np.percentile(d, 99)
    pts, cols = pts[keep], cols[keep]
    print(f"{len(pts)} initial points")

    with open(os.path.join(txt_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {Wo} {Ho} {f_orig:.6f} {f_orig:.6f} {cx:.3f} {cy:.3f}\n")
    with open(os.path.join(txt_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for k in range(n):
            x, y, z, w = Rot.from_matrix(R[k]).as_quat()
            f.write(f"{k + 1} {w:.9f} {x:.9f} {y:.9f} {z:.9f} {T[k][0]:.9f} {T[k][1]:.9f} {T[k][2]:.9f} 1 {names[k]}\n\n")
    with open(os.path.join(txt_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, c) in enumerate(zip(pts, cols)):
            f.write(f"{i + 1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 1.0\n")

    sparse0 = os.path.join(a.out, "sparse", "0")
    os.makedirs(sparse0)
    colmap = a.colmap_exe
    if not colmap:
        from run_sfm import find_colmap
        try:
            colmap = find_colmap(None)
        except SystemExit:
            colmap = ""
    done = False
    if colmap:
        r = subprocess.run([colmap, "model_converter", "--input_path", txt_dir, "--output_path", sparse0, "--output_type", "BIN"],
                           capture_output=True, text=True)
        done = r.returncode == 0 and os.path.isfile(os.path.join(sparse0, "images.bin"))
        print("binary model written with colmap model_converter" if done else f"model_converter failed ({r.returncode}); using the text model")
    if not done:
        for fn in ("cameras.txt", "images.txt", "points3D.txt"):
            shutil.copy(os.path.join(txt_dir, fn), os.path.join(sparse0, fn))
    print(f"dataset ready: {a.out}  (images: {n})")


if __name__ == "__main__":
    main()
