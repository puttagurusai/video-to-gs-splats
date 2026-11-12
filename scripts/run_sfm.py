"""Stage 3-4: COLMAP structure-from-motion (CPU build is fine) + undistortion.

Usage (from project root):
    python -I scripts/run_sfm.py [--frames frames] [--work colmap] [--camera-model SIMPLE_RADIAL]
                                 [--max-image-size 1600] [--max-features 6000] [--overlap 15] [--dry-run]

Speed/resource choices: sequential matching (video frames are ordered, so this is far cheaper than
exhaustive), capped feature count, capped image size. Refuses to overwrite an existing database.
Writes <work>/database.db, <work>/sparse/<n>/, and <work>/dense/{images,sparse} (undistorted, PINHOLE).
"""
import argparse
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_CPU_COLMAP = os.path.join(PROJECT_ROOT, "tools", "colmap-cpu", "bin", "colmap.exe")   # self-contained CPU-only build (4.2.1)
PROJECT_CUDA_COLMAP = os.path.join(PROJECT_ROOT, "tools", "colmap-cuda", "bin", "colmap.exe")  # its GPU path crashes on our laptop
DEFAULT_SIBLING_COLMAP = r"C:\me\proj\3dscene reconstruction\tools\colmap\bin\colmap.exe"  # CPU-only, read-only fallback


def find_colmap(cli_value):
    for cand in (cli_value, os.environ.get("COLMAP_EXE"), PROJECT_CPU_COLMAP, PROJECT_CUDA_COLMAP, shutil.which("colmap"),
                 DEFAULT_SIBLING_COLMAP):
        if cand and os.path.isfile(cand):
            return cand
    raise SystemExit("colmap executable not found. Pass --colmap-exe or set COLMAP_EXE.")


def gpu_works(colmap, frames_dir):
    """True only if this COLMAP build has CUDA AND a real GPU SIFT run succeeds here.
    (On this laptop the official 4.2.1 CUDA build aborts inside CUDA init, so we probe instead of assuming.)"""
    import tempfile
    out = subprocess.run([colmap, "help"], capture_output=True, text=True).stdout.lower()
    if "with cuda" not in out or "without gpu" in out:
        return False
    imgs = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png")))[:1]
    if not imgs:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "img"))
        shutil.copy(os.path.join(frames_dir, imgs[0]), os.path.join(tmp, "img"))
        r = subprocess.run([colmap, "feature_extractor", "--database_path", os.path.join(tmp, "p.db"),
                            "--image_path", os.path.join(tmp, "img"), "--FeatureExtraction.use_gpu", "1"],
                           capture_output=True)
        return r.returncode == 0


def run(cmd, dry):
    print(">", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if dry:
        return
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {cmd[1]}")


def _val(x):
    return x() if callable(x) else x


def model_stats(path):
    import pycolmap
    rec = pycolmap.Reconstruction(path)
    n_reg = _val(rec.num_reg_images)
    try:
        err = _val(rec.compute_mean_reprojection_error)
    except Exception:
        err = float("nan")
    return n_reg, _val(rec.num_points3D), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--work", default="colmap")
    ap.add_argument("--colmap-exe")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL",
                    help="SIMPLE_RADIAL is the robust default for phone video; OPENCV needs more views")
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--max-features", type=int, default=6000)
    ap.add_argument("--overlap", type=int, default=15, help="sequential matching window")
    ap.add_argument("--min-registered-frac", type=float, default=0.8)
    ap.add_argument("--mapper", choices=["global", "incremental"], default="global",
                    help="global = GLOMAP-style global_mapper (much faster); incremental = classic, more tolerant")
    ap.add_argument("--reuse-db", action="store_true",
                    help="skip feature extraction + matching and map the existing database.db")
    ap.add_argument("--focal-factor", type=float, default=0.69,
                    help="focal prior = factor * longer image side (0.69 ~ phone main camera, 26 mm-equivalent). "
                         "Needed by the global mapper; 0 disables. ASSUMPTION: adjust for ultrawide/zoomed video.")
    ap.add_argument("--ba-iterations", type=int, default=3, help="global mapper bundle-adjustment rounds (1 is fastest)")
    ap.add_argument("--fast", action="store_true",
                    help="preset for quick previews: 3000 features, 1200 px, overlap 8 (pair with extract_frames --window 6)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if the COLMAP build has CUDA")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.fast:
        a.max_features, a.max_image_size, a.overlap, a.ba_iterations = 3000, 1200, 8, 1
    colmap = find_colmap(a.colmap_exe)
    frames_dir = os.path.abspath(a.frames)
    gpu = "1" if (not a.cpu and os.path.isdir(frames_dir) and gpu_works(colmap, frames_dir)) else "0"
    print(f"GPU for SIFT extraction/matching: {'ON' if gpu == '1' else 'OFF (no working CUDA path, or --cpu)'}")
    frames = os.path.abspath(a.frames)
    work = os.path.abspath(a.work)
    n_frames = len([f for f in os.listdir(frames) if f.lower().endswith((".jpg", ".png"))]) if os.path.isdir(frames) else 0
    print(f"colmap: {colmap}\nframes: {n_frames} in {frames}")
    if n_frames < 30 and not a.dry_run:
        raise SystemExit("Fewer than 30 frames. Run extract_frames.py first.")

    db = os.path.join(work, "database.db")
    if a.reuse_db:
        if not os.path.exists(db):
            raise SystemExit(f"--reuse-db given but {db} does not exist.")
    elif os.path.exists(db) and not a.dry_run:
        raise SystemExit(f"{db} exists. Move or delete it deliberately, or pass --reuse-db.")
    sparse = os.path.join(work, "sparse")
    dense = os.path.join(work, "dense")
    os.makedirs(sparse, exist_ok=True)

    prior = []
    if a.focal_factor > 0 and a.camera_model == "SIMPLE_RADIAL" and n_frames:
        import cv2
        first = sorted(f for f in os.listdir(frames) if f.lower().endswith((".jpg", ".png")))[0]
        h0, w0 = cv2.imread(os.path.join(frames, first)).shape[:2]
        s = min(1.0, a.max_image_size / max(h0, w0))  # COLMAP downsizes to max_image_size; params are in that frame
        w1, h1 = w0 * s, h0 * s
        f = a.focal_factor * max(w1, h1)
        prior = ["--ImageReader.camera_params", f"{f:.2f},{w1 / 2:.2f},{h1 / 2:.2f},0"]
        print(f"focal prior: {f:.1f} px for {w1:.0f}x{h1:.0f} (factor {a.focal_factor})")
    if not a.reuse_db:
        run([colmap, "feature_extractor", "--database_path", db, "--image_path", frames,
             "--ImageReader.camera_model", a.camera_model, "--ImageReader.single_camera", "1"] + prior + [
             "--FeatureExtraction.use_gpu", gpu, "--FeatureExtraction.max_image_size", str(a.max_image_size),
             "--SiftExtraction.max_num_features", str(a.max_features)], a.dry_run)
        run([colmap, "sequential_matcher", "--database_path", db, "--FeatureMatching.use_gpu", gpu,
             "--SequentialMatching.overlap", str(a.overlap), "--SequentialMatching.loop_detection", "0"], a.dry_run)
    mapper_cmd = "global_mapper" if a.mapper == "global" else "mapper"
    extra = ["--GlobalMapper.ba_num_iterations", str(a.ba_iterations)] if a.mapper == "global" else []
    run([colmap, mapper_cmd, "--database_path", db, "--image_path", frames, "--output_path", sparse] + extra, a.dry_run)
    if a.dry_run:
        return 0

    models = sorted(d for d in os.listdir(sparse) if os.path.isdir(os.path.join(sparse, d)))
    if not models:
        raise SystemExit("GATE FAILED: mapper produced no model. Recapture with more overlap and parallax.")
    best, best_n = None, -1
    for m in models:
        n_reg, n_pts, err = model_stats(os.path.join(sparse, m))
        print(f"model {m}: {n_reg}/{n_frames} images, {n_pts} points, mean reproj err {err:.2f} px")
        if n_reg > best_n:
            best, best_n = m, n_reg
    frac = best_n / max(1, n_frames)
    print(f"Using model {best}: registered fraction {frac:.2f}")
    if len(models) > 1:
        print("WARNING: multiple models (fragmented reconstruction). Using the largest only.")
    if frac < a.min_registered_frac:
        raise SystemExit(f"GATE FAILED: only {frac:.0%} of frames registered (< {a.min_registered_frac:.0%}). "
                         "Recapture with more overlap/parallax rather than tuning.")

    run([colmap, "image_undistorter", "--image_path", frames, "--input_path", os.path.join(sparse, best),
         "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(a.max_image_size)], False)
    print(f"Done. Undistorted images + PINHOLE model in {dense}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
