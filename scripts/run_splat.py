"""One command: video -> Gaussian splat and/or depth-based textured mesh, with pose gates, a DA3 fallback, a time budget,
live progress, resume and a held-out quality check.

  python -I scripts/run_splat.py --video path\\to\\room.mp4 [--name room] [--mode splat|mesh|both] [--steps 4000]
(or double-click run_video_to_splat.bat, which holds every setting at the top)

Stages (each logged under runs/<name>/logs):
  1. frames    extract_frames.py: sharpest frame per window; --fps, capped by --max-frames but never below --min-fps
  2. poses     COLMAP + GLOMAP-style global mapper (run_sfm.py). GATES: enough frames registered AND sfm_check.py (not depth-flipped).
               If a gate fails, or the estimated COLMAP time exceeds --time-budget-min, or --poses da3: DA3 poses instead (faster, lower quality).
  3a. splat    Brush training (WebGPU, no CUDA toolchain); every Nth frame is held out; held-out SSIM/PSNR/coarse SSIM       (--mode splat|both)
  3b. mesh     depth-based reconstruction: depth alignment + refinement -> TSDF fusion -> UV unwrap -> blended texture -> .blend/.glb (--mode mesh|both)
  4. summary   runs/<name>/summary.json and how to view the results

Progress: the percentage is an ESTIMATE from typical stage durations (COLMAP and Brush report no percentage of their own).
Time estimates are measured on 125-142 frame clips on an RTX 500 (4 GB) laptop and EXTRAPOLATED for longer videos (rough).
Splats look much closer to the video (held-out coarse SSIM ~0.94 vs ~0.65 for the mesh on our mouse clip) but are not meshes.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "scripts")
PY = sys.executable
VENV_PY = os.path.join(ROOT, "tools", "venv_da3", "Scripts", "python.exe")
BRUSH = os.path.join(ROOT, "tools", "brush", "brush_app.exe")
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"


def mmss(sec):
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


class Progress:
    """Estimated overall progress: stages weighted by typical durations (an estimate, not a measurement)."""

    def __init__(self):
        self.plan, self.done, self.t0, self.last_print = {}, 0.0, time.time(), 0.0
        self.tty = sys.stdout.isatty()
        self.cols = max(60, shutil.get_terminal_size((110, 20)).columns - 1)
        self.dirty = False
        self.finished = False

    def percent(self, elapsed=0.0, expected=0.0):
        if self.finished:
            return 100.0
        cur = min(elapsed, 0.95 * expected) if expected else 0.0
        return min(99.0, 100.0 * (self.done + cur) / max(sum(self.plan.values()), 1.0))

    def show(self, label, elapsed, expected, last, force=False):
        now = time.time()
        if not force and now - self.last_print < (1.0 if self.tty else 20.0):
            return
        self.last_print = now
        pct = self.percent(elapsed, expected)
        filled = int(round(24 * pct / 100))
        line = f"[{'#' * filled}{'.' * (24 - filled)}] {pct:3.0f}%  {label}  {mmss(elapsed)}  (total {mmss(now - self.t0)})  {last}"
        if self.tty:
            print("\r" + line[:self.cols].ljust(self.cols), end="", flush=True)
            self.dirty = True
        else:
            print(line, flush=True)

    def newline(self):
        if self.tty and self.dirty:
            print(flush=True)
            self.dirty = False


PROG = Progress()


def kill_tree(p):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)


def last_line(path, limit=64):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4000))
            txt = f.read().decode("utf-8", errors="replace")
        lines = [re.sub(r"[\x00-\x1f]+", " ", l).strip() for l in re.split(r"[\r\n]+", txt)]
        lines = [l for l in lines if l]
        l = re.sub(r"^[IWEF]\d{8} [\d:.]+\s+\d+\s+", "", lines[-1]) if lines else ""
        return ("last: " + l[:limit]) if l else ""
    except OSError:
        return ""


def run(cmd, log, timeout=None, sample_vram=False, label="", expected=0.0):
    """Run a command, stream output to `log`, show live progress. Returns (returncode, seconds, peak_vram_MiB or None)."""
    t0, peak, last_sample = time.time(), 0, 0.0
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        rc = None
        while rc is None:
            try:
                rc = p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                el = time.time() - t0
                if sample_vram and time.time() - last_sample >= 2:
                    last_sample = time.time()
                    try:
                        m = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                               capture_output=True, text=True).stdout.strip().splitlines()[0])
                        peak = max(peak, m)
                    except Exception:
                        pass
                PROG.show(label, el, expected, last_line(log))
                if timeout and el > timeout:
                    kill_tree(p)
                    rc = -9
    PROG.done += expected
    return rc, time.time() - t0, (peak if sample_vram else None)


def tail(path, n=6):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).rstrip()
    except OSError:
        return ""


def video_info(video):
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        n, f = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if n > 0 and f > 0:
            return n / f, f
    except Exception:
        pass
    return None, None


def est_colmap(n):
    """COLMAP+global mapper seconds for n frames: ~1.4 s/frame measured at 125-142 frames, extrapolated superlinearly (rough)."""
    return 1.4 * n * max(1.0, n / 150.0) ** 0.5


def build_plan(n, a, route):
    """Expected stage durations in seconds (only used for the percentage and for the time budget)."""
    plan = {"frames": 8.0 + 0.02 * n}
    if route == "colmap":
        plan.update({"colmap": est_colmap(n), "flip": 25.0})
    else:
        plan.update({"da3": 0.4 * n, "export": 0.1 * n + 5.0})
    if a.mode in ("splat", "both"):
        plan.update({"brush": 10.0 + 0.02 * a.steps + 0.05 * n, "eval": 3.0})
    if a.mode in ("mesh", "both"):
        if route == "colmap":
            plan.update({"align": 1.0 * n, "refine": 0.15 * n + 5.0, "fuse": 25.0})
        else:
            plan.update({"da3_mesh": 0.5 * n + 30.0})
        plan.update({"uv": 8.0, "texture": 45.0, "blend": 8.0})
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--mode", choices=["splat", "mesh", "both"], default="splat", help="splat = Gaussian splat (Brush); mesh = depth-based textured mesh; both")
    # frames
    ap.add_argument("--fps", type=float, default=10.0, help="target kept frames per second of video")
    ap.add_argument("--max-frames", type=int, default=400, help="soft cap on frames (bounds COLMAP time)")
    ap.add_argument("--min-fps", type=float, default=3.0, help="never go below this many frames/s even if that exceeds --max-frames (SfM needs overlap)")
    # splat training
    ap.add_argument("--steps", type=int, default=4000, help="Brush training steps (4000 = good trade-off; quality gains flatten after)")
    ap.add_argument("--max-res", type=int, default=800, help="Brush training resolution (longer side, px)")
    ap.add_argument("--max-splats", type=int, default=600000)
    ap.add_argument("--heldout-every", type=int, default=10)
    # poses / SfM (COLMAP + GLOMAP-style global mapper)
    ap.add_argument("--poses", choices=["auto", "colmap", "da3"], default="auto")
    ap.add_argument("--sfm-mapper", choices=["global", "incremental"], default="global", help="global = GLOMAP-style (fast); incremental = classic (slower, more tolerant)")
    ap.add_argument("--sfm-features", type=int, default=3000, help="max SIFT features per image")
    ap.add_argument("--sfm-image-size", type=int, default=1200, help="max image side for SIFT (px)")
    ap.add_argument("--sfm-overlap", type=int, default=8, help="sequential matching window (frames)")
    ap.add_argument("--sfm-ba-iterations", type=int, default=1, help="global-mapper bundle adjustment rounds")
    ap.add_argument("--sfm-camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--sfm-min-registered", type=float, default=0.8, help="minimum fraction of frames that must register")
    ap.add_argument("--sfm-gpu", choices=["auto", "off"], default="auto", help="try GPU SIFT/matching if the COLMAP build has CUDA and a probe succeeds")
    ap.add_argument("--colmap-exe", default="", help="COLMAP executable (default: tools/colmap-cpu, then tools/colmap-cuda)")
    ap.add_argument("--focal-factor", type=float, default=0.72, help="COLMAP focal prior as a fraction of the longer side")
    ap.add_argument("--sfm-timeout", type=int, default=1800, help="seconds before COLMAP is abandoned")
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="if >0 and --poses auto: use the faster DA3 route when the estimated total exceeds this many minutes")
    # misc
    ap.add_argument("--blender", default=DEFAULT_BLENDER, help="Blender executable (needed for --mode mesh|both)")
    ap.add_argument("--force", action="store_true", help="reuse an existing run folder")
    ap.add_argument("--resume", action="store_true", help="continue an earlier run of the same name: skip stages whose outputs already exist")
    ap.add_argument("--open-viewer", action="store_true", help="open the finished splat in Brush's viewer")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and time estimate, then exit")
    ap.add_argument("--assume-duration", type=float, default=0.0, help="with --plan-only: pretend the video lasts this many seconds")
    a = ap.parse_args()

    a.video = os.path.abspath(a.video)
    a.out_root = os.path.abspath(a.out_root)
    if not os.path.isfile(a.video) and not (a.plan_only and a.assume_duration):
        raise SystemExit(f"video not found: {a.video}")
    if a.mode in ("splat", "both") and not os.path.isfile(BRUSH) and not a.plan_only:
        raise SystemExit(f"Brush not found at {BRUSH} (see RUNLOG.md for the verified download)")
    if a.mode in ("mesh", "both") and not os.path.isfile(a.blender) and not a.plan_only:
        raise SystemExit(f"Blender not found at {a.blender}: set BLENDER in the bat file (needed for the mesh mode)")

    # frame budget: honour --fps and --max-frames, but never fall below --min-fps
    dur, srcfps = (a.assume_duration, 30.0) if (a.plan_only and a.assume_duration) else video_info(a.video)
    eff_fps, fps_note = a.fps, ""
    if dur:
        if dur * a.fps > a.max_frames:
            eff_fps = a.max_frames / dur
        if eff_fps < a.min_fps:
            eff_fps = a.min_fps
            fps_note = (f"long video ({mmss(dur)}): using {eff_fps:g} fps = about {int(dur * eff_fps)} frames (above the {a.max_frames}-frame cap) "
                        f"because fewer frames would not overlap enough for SfM")
        elif eff_fps < a.fps:
            fps_note = f"video of {mmss(dur)}: lowering the frame rate to {eff_fps:.2f} fps to stay within {a.max_frames} frames"
        est_n = int(round(dur * min(eff_fps, srcfps or eff_fps)))
    else:
        est_n = a.max_frames // 2

    route = "da3" if a.poses == "da3" else "colmap"
    plan = build_plan(est_n, a, route)
    switched = ""
    if a.poses == "auto" and a.time_budget_min > 0 and sum(plan.values()) > a.time_budget_min * 60:
        route = "da3"
        plan = build_plan(est_n, a, route)
        switched = (f"estimated {mmss(sum(build_plan(est_n, a, 'colmap').values()))} with COLMAP exceeds the {a.time_budget_min:g}-minute budget: "
                    f"using the faster DA3 poses (lower quality)")
    print(f"Plan: mode={a.mode}  frames~{est_n}  poses route={route}  estimated total ~{mmss(sum(plan.values()))} (rough extrapolation)")
    for k, v in plan.items():
        print(f"   {k:9s} ~{mmss(v)}")
    for n_ in (fps_note, switched):
        if n_:
            print("NOTE: " + n_)
    if a.plan_only:
        return

    name = a.name or re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(a.video))[0])
    run_dir = os.path.join(a.out_root, name)
    if os.path.exists(run_dir) and not (a.force or a.resume):
        raise SystemExit(f"{run_dir} exists. Pick another NAME, or set RESUME=1 (continue it) or FORCE=1 (reuse the folder).")
    logs = os.path.join(run_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    frames_dir, colmap_dir = os.path.join(run_dir, "frames"), os.path.join(run_dir, "colmap")
    T, notes, T0 = {}, [], time.time()
    summary = dict(video=a.video, name=name, settings=vars(a), timings_s=T, notes=notes)
    for n_ in (fps_note, switched):
        if n_:
            notes.append(n_)

    def fail(msg):
        PROG.newline()
        summary["error"] = msg
        json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1, default=str)
        print(f"\nFAILED: {msg}\nlogs: {logs}")
        raise SystemExit(1)

    def step(label):
        PROG.newline()
        print(f"[{time.time() - T0:5.0f}s] {label}", flush=True)

    def skip(label, key):
        PROG.done += PROG.plan.get(key, 0.0)
        print(f"   (resume) skipping {label}: output already exists", flush=True)

    def done(path):
        return a.resume and os.path.exists(path)

    PROG.plan = dict(plan)
    print(f"\nRun folder: {run_dir}\nLogs:       {logs}\nProgress % is an estimate from typical stage durations.\n", flush=True)

    # 1. frames
    step("1/4 extracting frames")
    jpgs = lambda: len([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]) if os.path.isdir(frames_dir) else 0
    if done(frames_dir) and jpgs() >= 20:
        skip("frame extraction", "frames")
    else:
        rc, t, _ = run([PY, "-I", os.path.join(S, "extract_frames.py"), a.video, frames_dir, "--fps", f"{eff_fps:.4f}", "--max-side", "1200"],
                       os.path.join(logs, "1_frames.log"), label="extracting frames", expected=PROG.plan["frames"])
        T["frames"] = round(t, 1)
        if rc != 0 or jpgs() < 20:
            fail(f"frame extraction failed or too few frames ({jpgs()}); see {os.path.join(logs, '1_frames.log')}\n{tail(os.path.join(logs, '1_frames.log'))}")
    nframes = jpgs()
    summary["frames"] = nframes
    if nframes != est_n:  # real count known: refresh the estimates
        PROG.plan = {k: v for k, v in build_plan(nframes, a, route).items()}

    # 2. poses
    dataset, pose_source = None, None
    dense = os.path.join(colmap_dir, "dense")
    if route == "colmap":
        step(f"2/4 COLMAP + {a.sfm_mapper} mapper on {nframes} frames (timeout {mmss(a.sfm_timeout)})")
        if done(os.path.join(dense, "sparse", "images.bin")):
            skip("COLMAP", "colmap")
            rc = 0
        else:
            cmd = [PY, "-I", os.path.join(S, "run_sfm.py"), "--frames", frames_dir, "--work", colmap_dir, "--mapper", a.sfm_mapper,
                   "--max-features", str(a.sfm_features), "--max-image-size", str(a.sfm_image_size), "--overlap", str(a.sfm_overlap),
                   "--ba-iterations", str(a.sfm_ba_iterations), "--camera-model", a.sfm_camera_model,
                   "--min-registered-frac", str(a.sfm_min_registered), "--focal-factor", str(a.focal_factor)]
            if a.sfm_gpu == "off":
                cmd.append("--cpu")
            if a.colmap_exe:
                cmd += ["--colmap-exe", a.colmap_exe]
            rc, t, _ = run(cmd, os.path.join(logs, "2_colmap.log"), timeout=a.sfm_timeout, label="COLMAP + global mapper", expected=PROG.plan["colmap"])
            T["colmap"] = round(t, 1)
        if rc == 0 and os.path.isdir(os.path.join(dense, "sparse")):
            step("   gate: flip check (sparse depth vs monocular depth)")
            flog = os.path.join(logs, "2b_flipcheck.log")
            if a.resume and '"passed": true' in tail(flog, 1):
                skip("flip check", "flip")
                rc2 = 0
            else:
                rc2, t2, _ = run([PY, "-I", os.path.join(S, "sfm_check.py"), "--work", dense], flog, label="flip check", expected=PROG.plan["flip"])
                T["flip_check"] = round(t2, 1)
            chk = tail(flog, 1)
            summary["flip_check"] = chk
            if rc2 == 0:
                dataset, pose_source = dense, "colmap/glomap"
            else:
                notes.append(f"COLMAP model REJECTED by the flip check: {chk}")
        else:
            why = "timed out" if rc == -9 else "registration gate failed or error"
            notes.append(f"COLMAP did not produce a usable model ({why}): {tail(os.path.join(logs, '2_colmap.log'), 2)}")
        if dataset is None:
            PROG.newline()
            print("   " + notes[-1].replace("\n", " | "), flush=True)
    if dataset is None and a.poses == "colmap":
        fail("COLMAP route requested but it did not pass its gates (see notes in summary.json)")
    if dataset is None:
        PROG.plan.update({"da3": 0.4 * nframes, "export": 0.1 * nframes + 5.0})   # fallback adds work: the percentage estimate is extended
        step("2/4 DA3 poses (less accurate than COLMAP; expect lower quality)")
        if not os.path.isfile(VENV_PY):
            fail("DA3 environment missing (tools/venv_da3)")
        da3_dir = os.path.join(run_dir, "da3")
        bundle = os.path.join(da3_dir, "scene_cams.npz")
        if done(bundle):
            skip("DA3 poses", "da3")
        else:
            rc, t, _ = run([VENV_PY, "-u", "-I", os.path.join(S, "da3_reconstruct.py"), "--frames", frames_dir, "--out", os.path.join(da3_dir, "scene.ply"),
                            "--poses-only", "--focal", "da3"], os.path.join(logs, "2c_da3.log"), label="DA3 poses", expected=PROG.plan["da3"])
            T["da3"] = round(t, 1)
            if rc != 0 or not os.path.isfile(bundle):
                fail(f"DA3 pose estimation failed\n{tail(os.path.join(logs, '2c_da3.log'))}")
        dataset = os.path.join(run_dir, "dataset")
        if done(os.path.join(dataset, "sparse", "0", "images.bin")):
            skip("dataset export", "export")
        else:
            rc, t, _ = run([PY, "-I", os.path.join(S, "export_colmap_dataset.py"), "--cams", bundle, "--out", dataset, "--focal", "auto"],
                           os.path.join(logs, "2d_export.log"), label="exporting dataset", expected=PROG.plan["export"])
            T["export"] = round(t, 1)
            if rc != 0:
                fail(f"dataset export failed\n{tail(os.path.join(logs, '2d_export.log'))}")
        pose_source = "da3 (fallback)"
    summary["pose_source"] = pose_source

    ply, peak, m, ssim, psnr, bssim, nf = None, None, None, 0, 0, 0, 0
    # 3a. splat
    if a.mode in ("splat", "both"):
        splat_dir = os.path.join(run_dir, "splat")
        os.makedirs(splat_dir, exist_ok=True)
        ply = os.path.join(splat_dir, f"export_{a.steps}.ply")
        step(f"3/4 Brush training ({a.steps} steps, {a.max_res}px, <= {a.max_splats} splats) on {pose_source} poses")
        if done(ply):
            skip("Brush training", "brush")
        else:
            cmd = [BRUSH, dataset, "--total-steps", str(a.steps), "--max-resolution", str(a.max_res), "--max-splats", str(a.max_splats),
                   "--sh-degree", "2", "--eval-split-every", str(a.heldout_every), "--eval-every", str(a.steps), "--eval-save-to-disk",
                   "--export-path", splat_dir, "--export-every", str(a.steps)]
            rc, t, peak = run(cmd, os.path.join(logs, "3_brush.log"), sample_vram=True, timeout=3600, label=f"Brush training ({a.steps} steps)", expected=PROG.plan["brush"])
            T["brush"], summary["peak_gpu_mem_MiB"] = round(t, 1), peak
            if not os.path.isfile(ply):
                fail(f"Brush produced no splat (exit {rc}); see {os.path.join(logs, '3_brush.log')}\n{tail(os.path.join(logs, '3_brush.log'))}")
        summary["splat"] = ply
        step("   scoring held-out frames")
        rc, t, _ = run([PY, "-I", os.path.join(S, "eval_splat.py"), "--renders", os.path.join(splat_dir, f"eval_{a.steps}"), "--frames", os.path.join(dataset, "images")],
                       os.path.join(logs, "4_eval.log"), label="scoring", expected=PROG.plan["eval"])
        T["eval"] = round(t, 1)
        m = re.search(r"MEAN\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\((\d+) frames\)", tail(os.path.join(logs, "4_eval.log"), 3))
        if m:
            ssim, psnr, bssim, nf = float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
            summary["heldout"] = dict(ssim=ssim, psnr_db=psnr, coarse_ssim=bssim, frames=nf)
            summary["verdict"] = "GOOD" if ssim >= 0.80 else ("FAIR" if ssim >= 0.60 else "POOR")
        else:
            summary["verdict"] = "UNKNOWN (evaluation did not run)"

    # 3b. depth-based mesh
    mesh_blend = None
    if a.mode in ("mesh", "both"):
        step(f"3/4 depth-based mesh reconstruction on {pose_source} poses")
        mdir = os.path.join(run_dir, "mesh")
        os.makedirs(mdir, exist_ok=True)
        scene, cams = os.path.join(mdir, "scene.ply"), os.path.join(mdir, "scene_cams.npz")
        uv_obj, tex_obj = os.path.join(mdir, "scene_uv.obj"), os.path.join(mdir, "scene_tex.obj")
        blend, glb = os.path.join(mdir, "scene.blend"), os.path.join(mdir, "scene.glb")

        def stage(key, label, cmd, out_check, log):
            if done(out_check):
                skip(label, key)
                return True
            rc_, t_, _ = run(cmd, os.path.join(logs, log), label=label, expected=PROG.plan.get(key, 10.0))
            T[key] = round(t_, 1)
            ok_ = rc_ == 0 and os.path.exists(out_check)
            if not ok_:
                notes.append(f"mesh stage '{label}' failed; see {os.path.join(logs, log)}: {tail(os.path.join(logs, log), 2)}")
                PROG.newline()
                print("   " + notes[-1].replace("\n", " | "), flush=True)
            return ok_

        ok = True
        if pose_source == "colmap/glomap":
            aligned, refined = os.path.join(mdir, "aligned.npz"), os.path.join(mdir, "refined.npz")
            ok = ok and stage("align", "depth alignment (spec)", [PY, "-I", os.path.join(S, "align_real.py"), "--work", dense, "--out", aligned], aligned, "m1_align.log")
            ok = ok and stage("refine", "multi-view depth refinement", [PY, "-I", os.path.join(S, "refine_depth.py"), "--cams", aligned, "--out", refined,
                                                                         "--range", "0.97", "1.03", "--hyp", "25"], refined, "m2_refine.log")
            ok = ok and stage("fuse", "TSDF fusion + meshing", [PY, "-I", os.path.join(S, "fuse_aligned.py"), "--aligned", refined, "--out", scene], cams, "m3_fuse.log")
        else:
            ok = ok and stage("da3_mesh", "DA3 depth + fusion", [VENV_PY, "-u", "-I", os.path.join(S, "da3_reconstruct.py"), "--frames", frames_dir, "--out", scene,
                                                                "--focal", "auto"], cams, "m3_da3.log")
        ok = ok and stage("uv", "UV unwrap (Blender)", [a.blender, "--background", "--python", os.path.join(S, "uv_unwrap.py"), "--", "--input", scene,
                                                         "--output", uv_obj, "--target-tris", "150000"], uv_obj, "m4_uv.log")
        ok = ok and stage("texture", "blended texture", [PY, "-I", os.path.join(S, "texture_blend.py"), "--mesh", uv_obj, "--cams", cams, "--out", tex_obj], tex_obj, "m5_texture.log")
        ok = ok and stage("blend", "Blender file", [a.blender, "--background", "--python", os.path.join(S, "obj_to_blend.py"), "--", "--input", tex_obj,
                                                    "--blend", blend, "--glb", glb, "--name", "Scene"], blend, "m6_blend.log")
        if ok:
            mesh_blend = blend
            summary["mesh_blend"], summary["mesh_glb"] = blend, glb
        elif a.mode == "mesh":
            fail("mesh reconstruction failed (see notes in summary.json and logs)")

    T["total"] = round(time.time() - T0, 1)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1, default=str)
    PROG.finished = True
    PROG.show("finished", 0, 0, "", force=True)
    PROG.newline()
    print("\n=== SUMMARY ===")
    print(f"video:        {a.video}")
    print(f"mode:         {a.mode}   frames: {nframes}   poses: {pose_source}")
    for n_ in notes:
        print(f"note:         {n_.splitlines()[0]}")
    print("time (s):     " + ", ".join(f"{k} {v}" for k, v in T.items()))
    if peak:
        print(f"peak GPU mem: {peak} MiB")
    if ply:
        if m:
            print(f"held-out:     SSIM {ssim:.3f}  PSNR {psnr:.1f} dB  coarse SSIM {bssim:.3f}  ({nf} frames never trained on)")
        print(f"verdict:      {summary.get('verdict')}" + ("" if pose_source == "colmap/glomap" else "   (DA3 poses: quality is usually lower; consider re-capturing with more parallax)"))
        print(f"splat:        {ply}")
        print(f"view it:      tools\\brush\\brush_app.exe \"{ply}\"")
    if mesh_blend:
        print(f"mesh (.blend): {mesh_blend}   (depth-based textured mesh; lower visual quality than the splat)")
    print(f"summary:      {os.path.join(run_dir, 'summary.json')}")
    if a.open_viewer and ply:
        print("opening the splat in Brush's viewer ...", flush=True)
        subprocess.Popen([BRUSH, ply], cwd=ROOT)


if __name__ == "__main__":
    main()
