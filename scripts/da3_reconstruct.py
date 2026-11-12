"""Depth Anything 3 backend: feed-forward poses + intrinsics + depth, stitched in chunks, fused to a mesh.

Why: classical SfM failed on short, low-parallax phone clips (fragments / flipped depth), and depth-lifted PnP
cannot estimate focal length. DA3 predicts camera poses, intrinsics and multi-view-consistent depth in one pass.

Steps (run with the DA3 venv python, which also sees the system PyTorch/transformers):
  1. Frames are processed in overlapping chunks that fit in 4 GB (chunk size auto-probed against OOM).
  2. Each chunk is aligned to the previous one with a Sim3 (Umeyama, trimmed) fitted on back-projected depth
     points of the shared frames, giving one trajectory. Alignment residuals are reported.
  3. Absolute scale: DA3 (any-view) depth is up to an unknown scale, so the cached Depth-Anything-V2 *metric*
     model is compared with it per frame (median ratio). Spread is reported as a quality gate.
  4. DA3 depth (multi-view consistent) is fused with the shared GPU TSDF code (fuse.py) -> mesh.

Usage (project root):
  tools\\venv_da3\\Scripts\\python.exe -I scripts\\da3_reconstruct.py --frames frames_10fps --out export\\room_da3.ply

Licence note: use only Apache-2.0 DA3 checkpoints (DA3-SMALL / DA3-BASE). Larger/Nested ones are CC BY-NC.
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "da3_src", "src"))


def umeyama(src, dst, trim=0.8, iters=3):
    """Sim3 (s, R, t) with dst ~ s R src + t, refined on the best `trim` fraction of correspondences."""
    import numpy as np
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        S, D = src[keep], dst[keep]
        ms, md = S.mean(0), D.mean(0)
        Sc, Dc = S - ms, D - md
        U, sig, Vt = np.linalg.svd(Dc.T @ Sc / len(S))
        d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
        E = np.diag([1, 1, d])
        R = U @ E @ Vt
        s = (sig * np.diag(E)).sum() / ((Sc ** 2).sum() / len(S))
        t = md - s * R @ ms
        res = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
        keep = res <= np.quantile(res, trim)
    return s, R, t, float(np.median(res))


def grid_world_points(depth, conf, R, t, K, step=8, conf_q=50):
    """World points on a fixed pixel grid, plus a per-grid-pixel validity mask (depth > 0, confidence above the
    frame's own percentile). The grid is identical for any two views of the same frame, so the caller can
    intersect masks and get exact pixel-to-pixel correspondences."""
    import numpy as np
    H, W = depth.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    d = depth[ys, xs].astype(np.float64)
    valid = (d > 0) & (conf[ys, xs] >= np.percentile(conf, conf_q))
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    Xc = np.stack([(xs - K[0, 2]) / K[0, 0] * d, (ys - K[1, 2]) / K[1, 1] * d, d], -1).reshape(-1, 3)
    return (Xc - t) @ R, valid.ravel()  # world = R^T (Xc - t)


def run_chunk(model, paths, a):
    import numpy as np
    p = model.inference(paths, process_res=a.process_res, process_res_method="upper_bound_resize",
                        ref_view_strategy=a.ref_view)
    ext = np.asarray(p.extrinsics, dtype=np.float64)           # (m,3,4) world->cam (OpenCV)
    return dict(depth=np.asarray(p.depth, dtype=np.float32), conf=np.asarray(p.conf, dtype=np.float32),
                R=ext[:, :, :3], t=ext[:, :, 3], K=np.asarray(p.intrinsics, dtype=np.float64),
                rgb=np.asarray(p.processed_images))


def probe_chunk_size(model, paths, a):
    import torch
    for size in (24, 16, 12, 8, 6, 4):
        if size > a.chunk_max:
            continue
        try:
            run_chunk(model, paths[:size], a)
            torch.cuda.empty_cache()
            return size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  chunk {size} ran out of GPU memory, trying smaller", flush=True)
    raise SystemExit("Even 4 frames do not fit in GPU memory; lower --process-res.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--out", default="export/room_da3.ply")
    ap.add_argument("--model-id", default=os.path.join(ROOT, "tools", "da3_models", "DA3-BASE"),
                    help="local folder (downloaded with local_dir; Windows without symlink rights cannot use the HF cache)")
    ap.add_argument("--metric-model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    ap.add_argument("--online", action="store_true", help="allow Hugging Face downloads")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--chunk", type=int, default=0, help="frames per chunk (0 = auto-probe the largest that fits)")
    ap.add_argument("--chunk-max", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=5)
    ap.add_argument("--ref-view", default="saddle_balanced")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--conf-percentile", type=float, default=25.0, help="drop the least confident depth pixels")
    ap.add_argument("--max-depth", type=float, default=10.0, help="meters")
    ap.add_argument("--edge-grad", type=float, default=0.3)
    ap.add_argument("--consist-k", type=int, default=3)
    ap.add_argument("--consist-tol", type=float, default=0.08)
    ap.add_argument("--consist-min", type=int, default=2)
    ap.add_argument("--no-consist", action="store_true")
    ap.add_argument("--poses-only", action="store_true", help="write only the camera bundle (no TSDF/mesh); used by run_splat.py")
    ap.add_argument("--focal", default="auto",
                    help="'auto' = square pixels, focal chosen by multi-view photometric agreement; 'da3' = keep DA3's "
                         "fx/fy as predicted; or a number = focal / longer side")
    ap.add_argument("--min-frame-agree", type=float, default=0.3,
                    help="drop frames whose depth agrees with neighbours on fewer than this share of pixels")
    ap.add_argument("--scale-mult", type=float, default=1.0,
                    help="multiply the estimated metric scale (measure a real dimension in the result, "
                         "then real/measured gives this factor). Monocular metric scale is unreliable on close-ups.")
    ap.add_argument("--ceiling-height", type=float, default=0.0, help="REAL floor-to-ceiling height (m), optional")
    ap.add_argument("--voxel", default="auto", help="'auto' (from scene extent) or meters")
    ap.add_argument("--trunc-voxels", type=float, default=4.0)
    ap.add_argument("--max-voxels", type=int, default=30_000_000)
    ap.add_argument("--min-weight", type=float, default=0.2)
    ap.add_argument("--min-component-frac", type=float, default=0.02)
    a = ap.parse_args()
    if not a.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import cv2
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3
    from dense_tsdf import predict_depths
    from depth_odometry import consistency_filter
    from fuse import fuse_mesh_export

    T0 = time.time()
    paths = sorted(glob.glob(os.path.join(a.frames, "*.jpg")) + glob.glob(os.path.join(a.frames, "*.png")))[::a.stride]
    if a.max_frames and len(paths) > a.max_frames:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, a.max_frames).round().astype(int)]
    n = len(paths)
    if n < 6:
        raise SystemExit(f"Only {n} frames in {a.frames}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{n} frames; device={device}; loading {a.model_id}", flush=True)
    model = DepthAnything3.from_pretrained(a.model_id).to(device=device)
    model.eval()

    chunk = a.chunk or probe_chunk_size(model, paths, a)
    chunk = max(chunk, a.overlap + 2)
    print(f"[{time.time() - T0:.0f}s] chunk size {chunk}, overlap {a.overlap}, "
          f"peak GPU mem so far {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)

    spans, s0 = [], 0
    while True:
        e0 = min(n, s0 + chunk)
        spans.append((s0, e0))
        if e0 == n:
            break
        s0 = e0 - a.overlap

    # ---- chunked inference + Sim3 stitching into the first chunk's frame ----
    G = dict(R=[None] * n, t=[None] * n, depth=[None] * n, conf=[None] * n, K=[None] * n, rgb=[None] * n)
    align_err, chunk_scales = [], []
    for ci, (s, e) in enumerate(spans):
        out = run_chunk(model, paths[s:e], a)
        torch.cuda.empty_cache()
        if ci == 0:
            sc, Ra, ta = 1.0, np.eye(3), np.zeros(3)
        else:
            src, dst = [], []
            for k in range(s, s + a.overlap):  # frames already stored in the global frame
                i = k - s
                pn, vn = grid_world_points(out["depth"][i], out["conf"][i], out["R"][i], out["t"][i], out["K"][i])
                pg, vg = grid_world_points(G["depth"][k], G["conf"][k], G["R"][k], G["t"][k], G["K"][k])
                both = vn & vg  # pixel-to-pixel correspondences
                if both.any():
                    src.append(pn[both]); dst.append(pg[both])
            if not src:
                raise SystemExit("Chunk overlap has no usable depth points; increase --overlap.")
            sc, Ra, ta, err = umeyama(np.concatenate(src), np.concatenate(dst))
            align_err.append(err)
            chunk_scales.append(sc)
        for i in range(e - s):
            k = s + i
            if G["R"][k] is not None:
                continue  # keep the earlier chunk's estimate for shared frames
            Rk, tk = out["R"][i], out["t"][i]
            Ck = -Rk.T @ tk
            Cg = sc * Ra @ Ck + ta
            Rg = Rk @ Ra.T
            G["R"][k], G["t"][k] = Rg, -Rg @ Cg
            G["depth"][k], G["conf"][k] = out["depth"][i] * sc, out["conf"][i]
            G["K"][k], G["rgb"][k] = out["K"][i], out["rgb"][i]
        print(f"[{time.time() - T0:.0f}s] chunk {ci + 1}/{len(spans)} frames {s}-{e - 1}"
              + (f", Sim3 scale {sc:.3f}, overlap residual {align_err[-1]:.4f}" if ci else ""), flush=True)
    del model
    torch.cuda.empty_cache()
    if align_err:
        print(f"chunk alignment residual (median of medians): {np.median(align_err):.4f} DA3 units", flush=True)

    # ---- absolute scale from the cached metric model ----
    views = [dict(rgb=G["rgb"][k], H=G["rgb"][k].shape[0], W=G["rgb"][k].shape[1]) for k in range(n)]
    predict_depths(views, a.metric_model)
    ratios = []
    for k in range(n):
        d3, dm, c = G["depth"][k], views[k]["depth_raw"], G["conf"][k]
        ok = (d3 > 0) & (dm > 0.2) & (c >= np.percentile(c, 50))
        if ok.sum() > 100:
            ratios.append(float(np.median(dm[ok] / d3[ok])))
    if len(ratios) < 5:
        raise SystemExit("Could not establish metric scale (too few valid frames).")
    scale = float(np.median(ratios)) * a.scale_mult
    spread = float(np.std(np.log(ratios)))
    print(f"[{time.time() - T0:.0f}s] metric scale {scale:.4f} m per DA3 unit; frame-to-frame spread {spread:.3f}"
          + ("  WARNING: large spread, depth scale unreliable" if spread > 0.15 else ""), flush=True)

    # ---- build fusion inputs in DA3 units (fuse applies the metric scale at the end) ----
    Kmed = np.median(np.stack(G["K"]), axis=0)
    H, W = G["depth"][0].shape
    good = []
    for k in range(n):
        d = G["depth"][k].astype(np.float32).copy()
        c = G["conf"][k]
        d[c < np.percentile(c, a.conf_percentile)] = 0
        gx, gy = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        d[(np.hypot(gx, gy) / np.maximum(d, 1e-6)) > a.edge_grad] = 0
        d[(d <= 0) | (d * scale > a.max_depth)] = 0
        good.append(dict(R=G["R"][k], t=G["t"][k], K=Kmed, W=W, H=H, rgb=G["rgb"][k], depth=d,
                         name=os.path.basename(paths[k])))
    # square pixels: DA3's fx and fy can differ by ~20%, which distorts the lifted geometry. Choose f photometrically.
    focal_info = dict(mode=a.focal, da3_fx=float(Kmed[0, 0]), da3_fy=float(Kmed[1, 1]))
    if a.focal != "da3":
        if a.focal == "auto":
            import photo_calib
            grays = np.stack([cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                              for p in paths])
            f_iso, table = photo_calib.calibrate_isotropic_focal(
                np.stack([g["depth"] for g in good]), np.stack([g["R"] for g in good]), np.stack([g["t"] for g in good]),
                float(Kmed[0, 2]), float(Kmed[1, 2]), grays, device)
            if f_iso is None:
                f_iso = float(np.sqrt(Kmed[0, 0] * Kmed[1, 1]))
                print("focal calibration had too few usable pairs; using the geometric mean of DA3's fx, fy", flush=True)
            else:
                focal_info["scan_best"] = f_iso
        else:
            f_iso = float(a.focal) * max(W, H)
        print(f"focal: DA3 native fx={Kmed[0, 0]:.1f} fy={Kmed[1, 1]:.1f} -> square pixels f={f_iso:.1f}px "
              f"({f_iso / max(W, H):.3f} x longer side)", flush=True)
        Kmed = np.array([[f_iso, 0, Kmed[0, 2]], [0, f_iso, Kmed[1, 2]], [0, 0, 1.0]])
        for g in good:
            g["K"] = Kmed
    kept_frac = None
    if not a.no_consist:
        kept_frac = consistency_filter(good, a.consist_k, a.consist_tol, a.consist_min, device.type)
        bad_frames = [g["name"] for g in good if g["kept_frac"] < a.min_frame_agree]
        if bad_frames and len(good) - len(bad_frames) >= 6:  # bad pose/depth: agrees with none of its neighbours
            print(f"dropping {len(bad_frames)} frame(s) that disagree with their neighbours: {bad_frames[:6]}", flush=True)
            good = [g for g in good if g["kept_frac"] >= a.min_frame_agree]
    pts = []
    for g in good:
        sub = g["depth"][::16, ::16]
        yy, xx = np.nonzero(sub > 0)
        z = sub[yy, xx].astype(np.float64)
        Xc = np.stack([(xx * 16.0 - Kmed[0, 2]) / Kmed[0, 0] * z, (yy * 16.0 - Kmed[1, 2]) / Kmed[1, 1] * z, z], 1)
        pts.append((Xc - g["t"]) @ g["R"])
    pts = np.concatenate(pts)
    fx_ratio = float(Kmed[0, 0] / max(W, H))
    print(f"estimated focal: {Kmed[0, 0]:.1f} px at {W}x{H} ({fx_ratio:.3f} x longer side)", flush=True)

    if a.poses_only:  # splat pipeline: only poses/depth/intrinsics are needed, skip meshing
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez_compressed(os.path.splitext(a.out)[0] + "_cams.npz", names=np.array([g["name"] for g in good]),
                            R=np.stack([g["R"] for g in good]), t=np.stack([g["t"] for g in good]), K=Kmed,
                            hw=np.array([H, W]), depth=np.stack([g["depth"] for g in good]).astype(np.float16),
                            world_to_mesh=np.eye(4), frames_dir=os.path.abspath(a.frames))
        print(f"[{time.time() - T0:.0f}s] poses-only: wrote camera bundle {os.path.splitext(a.out)[0]}_cams.npz", flush=True)
        return 0
    mesh = fuse_mesh_export(good, pts, scale, a, dict(method="da3", model=a.model_id, frames=n, chunk=chunk,
                                              overlap=a.overlap, metric_scale=scale, scale_spread=spread,
                                              focal_factor=fx_ratio, chunk_align_residual=float(np.median(align_err)) if align_err else None))
    print(f"[{time.time() - T0:.0f}s] done; peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)

    # ---- automatic quality report: the checks to read before trusting a result ----
    import json
    comp = mesh.metadata.get("quality", {})
    nz = [g["depth"][g["depth"] > 0][::50] for g in good]
    med_depth = float(np.median(np.concatenate([d for d in nz if len(d)])))
    rel_resid = float(np.median(align_err) / med_depth) if align_err else 0.0
    scale_range = (max(chunk_scales) / min(chunk_scales)) if chunk_scales else 1.0
    lc = comp.get("largest_component_frac")
    checks = [
        ("metric-scale spread across frames (std log ratio)", spread, "<= 0.15", spread <= 0.15),
        ("chunk overlap residual / median depth", rel_resid, "<= 0.03", rel_resid <= 0.03),
        ("chunk-to-chunk scale range (max/min)", scale_range, "<= 1.5", scale_range <= 1.5),
        ("depth pixels kept by cross-view filter", kept_frac, ">= 0.80", kept_frac is None or kept_frac >= 0.80),
        ("largest mesh component share of faces", lc, ">= 0.60", lc is None or lc >= 0.60),
    ]
    bad = [c for c in checks if not c[3]]
    verdict = "OK" if not bad else ("CHECK" if len(bad) <= 2 else "POOR")
    print("\nQUALITY REPORT")
    for name, val, rule, ok in checks:
        print(f"  [{'pass' if ok else 'FAIL'}] {name}: {'n/a' if val is None else round(float(val), 3)} (want {rule})")
    print(f"  VERDICT: {verdict}" + ("" if verdict == "OK" else
          "  (see README.md, Capture checklist: re-capture with more parallax, or try --chunk 16 --overlap 8)"), flush=True)
    # camera bundle for texturing: poses/intrinsics/filtered depth in DA3 units + DA3-world -> mesh-frame matrix
    np.savez_compressed(os.path.splitext(a.out)[0] + "_cams.npz",
                        names=np.array([g["name"] for g in good]),
                        R=np.stack([g["R"] for g in good]), t=np.stack([g["t"] for g in good]),
                        K=Kmed, hw=np.array([H, W]), depth=np.stack([g["depth"] for g in good]).astype(np.float16),
                        world_to_mesh=mesh.metadata["world_to_mesh"], frames_dir=os.path.abspath(a.frames))
    print(f"wrote camera bundle {os.path.splitext(a.out)[0]}_cams.npz", flush=True)
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as f:
        json.dump(dict(verdict=verdict, checks=[dict(name=c[0], value=c[1], want=c[2], ok=bool(c[3])) for c in checks]),
                  f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
