"""Refine DA3 camera poses with COLMAP's own triangulation + bundle adjustment (no SfM registration step).

COLMAP/GLOMAP can fail to REGISTER a low-parallax clip even though feature matching worked. Here the DA3 poses are the
starting point: 3D points are triangulated from the matches in the existing COLMAP database given those poses
(`point_triangulator`), then poses and points are bundle-adjusted (`bundle_adjuster`), repeated for --rounds.

  python -I scripts/refine_poses.py --dataset runs/hotel/dataset --database runs/hotel/colmap/database.db --out runs/hotel_ba/dataset

--dataset   output of export_colmap_dataset.py (images/ + sparse/0 with the DA3 poses and a square-pixel PINHOLE camera)
--database  COLMAP database with features + verified matches for the SAME image names (e.g. runs/<name>/colmap/database.db)
Writes <out>/images/ and <out>/sparse/0/ (binary model with the refined poses; intrinsics fixed unless --refine-focal).
Prints reprojection error before/after and how far the poses moved (after a similarity alignment, since BA has free gauge).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def colmap(exe, *args):
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def analyze(exe, model):
    rc, out = colmap(exe, "model_analyzer", "--path", model)
    keep = [l.split("]")[-1].strip() for l in out.splitlines()
            if re.search(r"Points:|Observations:|Mean track length|Mean reprojection error|Registered (images|frames)", l)]
    err = re.search(r"Mean reprojection error:\s*([\d.]+)", out)
    return (float(err.group(1)) if err else float("nan")), keep


def match_database_camera(db_path, model_dir):
    """COLMAP 4.x ties cameras to rigs: the database camera must be the same model/params as the input model's camera,
    otherwise point_triangulator aborts. Rewrite the (copied) database's camera row to match the model."""
    import sqlite3
    import pycolmap
    rec = pycolmap.Reconstruction(model_dir)
    cams = list(rec.cameras.values())
    if len(cams) != 1:
        raise SystemExit("expected a single shared camera in the input model")
    cam = cams[0]
    name = cam.model.name if hasattr(cam.model, "name") else str(cam.model).split(".")[-1]
    ids = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}
    if name not in ids:
        raise SystemExit(f"camera model {name} not supported here (PINHOLE / SIMPLE_PINHOLE only)")
    db = sqlite3.connect(db_path)
    n = db.execute("select count(*) from cameras").fetchone()[0]
    if n != 1:
        raise SystemExit(f"database has {n} cameras; expected 1")
    db.execute("update cameras set model=?, width=?, height=?, params=?, prior_focal_length=1",
               (ids[name], int(cam.width), int(cam.height), np.asarray(cam.params, dtype=np.float64).tobytes()))
    db.commit()
    db.close()


def build_input_model(exe, dataset, db_path, dst_bin):
    """Input model for point_triangulator: our poses, but image IDs taken from the database BY NAME (COLMAP 4.x compares
    the frame-to-image pairing of the model with the database and aborts if they differ), and no 3D points."""
    import sqlite3
    txt_in = os.path.join(dataset, "sparse_txt")
    if not os.path.isfile(os.path.join(txt_in, "images.txt")):
        raise SystemExit("dataset/sparse_txt/images.txt not found (use a dataset made by export_colmap_dataset.py)")
    db = sqlite3.connect(db_path)
    db_ids = {name: iid for iid, name in db.execute("select image_id, name from images")}
    db.close()
    rows = []
    for line in open(os.path.join(txt_in, "images.txt")):
        t = line.split()
        if line.startswith("#") or len(t) < 10 or not t[-1].lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if t[-1] not in db_ids:
            raise SystemExit(f"image {t[-1]} is not in the database")
        rows.append((db_ids[t[-1]], t))
    if len(rows) != len(db_ids):
        raise SystemExit(f"model has {len(rows)} images but the database has {len(db_ids)}")
    tmp_txt = dst_bin + "_txt"
    os.makedirs(tmp_txt)
    os.makedirs(dst_bin)
    shutil.copy(os.path.join(txt_in, "cameras.txt"), os.path.join(tmp_txt, "cameras.txt"))
    with open(os.path.join(tmp_txt, "images.txt"), "w") as f:
        f.write("# Image list (IDs follow the database)\n")
        for iid, t in sorted(rows):
            f.write(" ".join([str(iid)] + t[1:]) + "\n\n")
    open(os.path.join(tmp_txt, "points3D.txt"), "w").write("# empty\n")
    rc, out = colmap(exe, "model_converter", "--input_path", tmp_txt, "--output_path", dst_bin, "--output_type", "BIN")
    if rc != 0 or not os.path.isfile(os.path.join(dst_bin, "images.bin")):
        raise SystemExit(f"model_converter failed:\n{out[-800:]}")


def load_poses(model):
    import pycolmap
    rec = pycolmap.Reconstruction(model)
    out = {}
    for im in rec.images.values():
        pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        M = np.asarray(pose.matrix() if callable(pose.matrix) else pose.matrix, dtype=np.float64)
        out[im.name] = (M[:, :3], M[:, 3])
    return out


def pose_shift(a_model, b_model):
    """Camera-centre and rotation change between two models after a similarity alignment (Umeyama)."""
    A, B = load_poses(a_model), load_poses(b_model)
    names = sorted(set(A) & set(B))
    Ca = np.array([-A[n][0].T @ A[n][1] for n in names])
    Cb = np.array([-B[n][0].T @ B[n][1] for n in names])
    ma, mb = Ca.mean(0), Cb.mean(0)
    X, Y = Ca - ma, Cb - mb
    U, S, Vt = np.linalg.svd(Y.T @ X / len(names))
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    s = (S * np.array([1, 1, d])).sum() / (X ** 2).sum() * len(names)
    Ca_al = (s * (Ca - ma) @ Rm.T) + mb
    step = float(np.median(np.linalg.norm(np.diff(Cb, axis=0), axis=1)))
    cres = np.linalg.norm(Ca_al - Cb, axis=1)
    rot = []
    for n in names:                                   # rotation difference after applying the alignment rotation
        Rrel = B[n][0] @ (A[n][0] @ Rm.T).T
        rot.append(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    return dict(frames=len(names), centre_shift_vs_camera_step=float(np.mean(cres) / max(step, 1e-12)),
                max_centre_shift_vs_step=float(np.max(cres) / max(step, 1e-12)), rot_deg_mean=float(np.mean(rot)), rot_deg_max=float(np.max(rot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--refine-focal", action="store_true", help="also refine the focal length in bundle adjustment")
    ap.add_argument("--colmap-exe", default="")
    a = ap.parse_args()
    exe = a.colmap_exe
    if not exe:
        from run_sfm import find_colmap
        exe = find_colmap(None)
    src_model = os.path.join(a.dataset, "sparse", "0")
    src_images = os.path.join(a.dataset, "images")
    for p in (src_model, src_images, a.database):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} already exists; not overwriting.")
    os.makedirs(os.path.join(a.out, "sparse", "0"))
    shutil.copytree(src_images, os.path.join(a.out, "images"))
    tmp = os.path.join(a.out, "_tmp")
    os.makedirs(tmp)
    db_work = os.path.join(tmp, "database.db")            # the original database is never modified
    shutil.copy2(a.database, db_work)
    match_database_camera(db_work, src_model)
    tri_in = os.path.join(tmp, "tri_in")
    build_input_model(exe, a.dataset, db_work, tri_in)

    cur = tri_in
    last_valid = None
    print(f"start: DA3 poses ({len(os.listdir(src_images))} images); intrinsics {'refined' if a.refine_focal else 'fixed'}", flush=True)
    for r in range(1, a.rounds + 1):
        tri, ba = os.path.join(tmp, f"tri{r}"), os.path.join(tmp, f"ba{r}")
        os.makedirs(tri), os.makedirs(ba)
        rc, out = colmap(exe, "point_triangulator", "--database_path", db_work, "--image_path", os.path.join(a.out, "images"),
                         "--input_path", cur, "--output_path", tri, "--refine_intrinsics", "0")
        if rc != 0 or not os.path.isfile(os.path.join(tri, "points3D.bin")):
            raise SystemExit(f"point_triangulator failed (round {r}):\n{out[-1500:]}")
        e_tri, lines = analyze(exe, tri)
        print(f"round {r} triangulated with current poses: mean reprojection error {e_tri:.2f} px | " + " | ".join(lines), flush=True)
        rc, out = colmap(exe, "bundle_adjuster", "--input_path", tri, "--output_path", ba,
                         "--BundleAdjustment.refine_focal_length", "1" if a.refine_focal else "0",
                         "--BundleAdjustment.refine_principal_point", "0", "--BundleAdjustment.refine_extra_params", "0")
        if rc != 0 or not os.path.isfile(os.path.join(ba, "images.bin")):
            raise SystemExit(f"bundle_adjuster failed (round {r}):\n{out[-1500:]}")
        e_ba, lines = analyze(exe, ba)
        print(f"round {r} after bundle adjustment:              mean reprojection error {e_ba:.3g} px | " + " | ".join(lines), flush=True)
        # divergence guard: a bundle adjustment that blows up must not replace the last good model
        if not np.isfinite(e_ba) or e_ba > max(3.0, 2.0 * e_tri):
            print(f"round {r} DIVERGED (reprojection error {e_ba:.3g} px vs {e_tri:.2f} px before): discarding it, keeping round {r - 1 if last_valid else 0}", flush=True)
            break
        last_valid = ba
        keep = os.path.join(a.out, "rounds", f"ba{r}")
        shutil.copytree(ba, keep)
        try:
            ps = pose_shift(src_model, ba)
            print(f"   round {r} poses vs DA3 (similarity-aligned): centres moved {ps['centre_shift_vs_camera_step']:.2f}x the frame step "
                  f"(max {ps['max_centre_shift_vs_step']:.2f}x), rotation mean {ps['rot_deg_mean']:.2f} deg, max {ps['rot_deg_max']:.2f} deg", flush=True)
        except Exception as exc:
            print(f"   (pose-change diagnostics skipped: {exc})")
        cur = ba
    if last_valid is None:
        raise SystemExit("bundle adjustment diverged in every round; no refined model written")
    cur = last_valid
    for fn in os.listdir(cur):
        shutil.copy(os.path.join(cur, fn), os.path.join(a.out, "sparse", "0", fn))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"refined dataset ready: {a.out}")


if __name__ == "__main__":
    main()
