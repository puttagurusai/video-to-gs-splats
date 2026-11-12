"""Stage 5-7 (low-resource replacement for OpenMVS): metric depth + SfM alignment + TSDF fusion.

For each undistorted frame:
  1. Depth-Anything-V2-Metric-Indoor predicts depth (cached locally, runs on the GPU, offline by default).
  2. Depth is aligned to the COLMAP sparse points of that frame (robust scale+shift fit), so all frames
     share COLMAP's multi-view-consistent scale instead of drifting per frame.
  3. Aligned depth is fused into a TSDF volume on the GPU, then meshed with marching cubes.
The mesh is scaled to meters (median of the per-frame fits), rotated Z-up using the camera "down" vectors,
and shifted so the floor (1st percentile of height) sits at Z=0. Per-vertex colors are fused too.

Usage (from project root):
    python -I scripts/dense_tsdf.py --work colmap/dense --out export/room_raw.ply [--voxel 0.03]
                                    [--depth-side 800] [--max-frames 0] [--max-voxels 30000000]

Writes <out> (PLY mesh, vertex colors, meters, Z-up) and <out stem>.json (parameters, versions, stats, spawn).
Heuristics worth checking by eye: gravity direction (assumes the phone was held upright, not rolled) and
floor height (assumes the floor is the lowest 1% of the scan). Monocular depth is weak on mirrors/glass.
"""
import argparse
import json
import os
import sys
import time


def _val(x):
    return x() if callable(x) else x


def load_recon(work):
    import pycolmap
    for cand in (os.path.join(work, "sparse"), os.path.join(work, "sparse", "0")):
        if os.path.isfile(os.path.join(cand, "images.bin")) or os.path.isfile(os.path.join(cand, "images.txt")):
            return pycolmap.Reconstruction(cand)
    raise SystemExit(f"No COLMAP model found under {work}/sparse. Run run_sfm.py first.")


def gather_views(rec, img_dir, side, max_frames):
    import cv2
    import numpy as np
    items = sorted(rec.images.items(), key=lambda kv: kv[1].name)
    if max_frames and len(items) > max_frames:
        keep = np.linspace(0, len(items) - 1, max_frames).round().astype(int)
        items = [items[i] for i in keep]
    views = []
    for _, im in items:
        cam = rec.cameras[im.camera_id]
        M = np.asarray(_val(_val(im.cam_from_world).matrix), dtype=np.float64)
        R, t = M[:, :3], M[:, 3]
        p = np.asarray(cam.params, dtype=np.float64)
        if len(p) == 4:
            fx, fy, cx, cy = p
        elif len(p) == 3:
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            raise SystemExit("Expected PINHOLE/SIMPLE_PINHOLE cameras (use the undistorted model).")
        bgr = cv2.imread(os.path.join(img_dir, im.name))
        if bgr is None:
            print(f"skip (unreadable): {im.name}")
            continue
        h0, w0 = bgr.shape[:2]
        s = min(1.0, side / max(h0, w0))
        W, H = int(round(w0 * s)), int(round(h0 * s))
        if (W, H) != (w0, h0):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        sx, sy = W / cam.width, H / cam.height
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float64)
        xy, X = [], []
        for p2 in im.points2D:
            try:
                ok = _val(p2.has_point3D)
            except AttributeError:
                ok = p2.point3D_id != 2**64 - 1
            if ok:
                xy.append(p2.xy)
                X.append(rec.points3D[p2.point3D_id].xyz)
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2) * np.array([sx, sy])
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        z = (R @ X.T + t[:, None])[2] if len(X) else np.zeros(0)
        views.append(dict(name=im.name, R=R, t=t, K=K, W=W, H=H, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                          xy=xy, z=z))
    return views


def predict_depths(views, model_id):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import pipeline
    cuda = torch.cuda.is_available()
    kw = dict(torch_dtype=torch.float16) if cuda else {}
    pipe = pipeline("depth-estimation", model=model_id, device=0 if cuda else -1, **kw)
    t0 = time.time()
    for i, v in enumerate(views):
        out = pipe(Image.fromarray(v["rgb"]))["predicted_depth"]
        d = out.squeeze().float().cpu().numpy() if torch.is_tensor(out) else np.asarray(out, dtype=np.float32).squeeze()
        v["depth_raw"] = np.clip(cv2.resize(d, (v["W"], v["H"]), interpolation=cv2.INTER_LINEAR), 0, None).astype(np.float32)
        if (i + 1) % 25 == 0 or i + 1 == len(views):
            print(f"  depth {i + 1}/{len(views)}  ({time.time() - t0:.0f}s)", flush=True)
    del pipe
    if cuda:
        torch.cuda.empty_cache()


def robust_fit(d, z, iters=10):
    """z ~ a*d + b via IRLS with Huber-like weights. Returns (a, b, relative residual)."""
    import numpy as np
    A = np.stack([d, np.ones_like(d)], 1)
    w = np.ones_like(d)
    sol = np.array([1.0, 0.0])
    for _ in range(iters):
        sol = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]
        r = z - A @ sol
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        w = 1.0 / np.maximum(1.0, np.abs(r) / (1.5 * s))
    r = z - A @ sol
    return sol[0], sol[1], float(np.median(np.abs(r)) / (np.median(z) + 1e-9))


def align_depths(views, min_pts, max_rel_resid):
    import cv2
    import numpy as np
    good, scales = [], []
    why = dict(few_points=0, negative_scale=0, high_residual=0)
    for v in views:
        v["ok"] = False
        if len(v["z"]) < min_pts:
            why["few_points"] += 1
            continue
        xi = np.clip(np.round(v["xy"][:, 0]).astype(int), 0, v["W"] - 1)
        yi = np.clip(np.round(v["xy"][:, 1]).astype(int), 0, v["H"] - 1)
        d = v["depth_raw"][yi, xi].astype(np.float64)
        m = (d > 0.1) & (v["z"] > 0)
        if m.sum() < min_pts:
            why["few_points"] += 1
            continue
        a, b, rel = robust_fit(d[m], v["z"][m])
        if a <= 0:
            why["negative_scale"] += 1
            continue
        if rel > max_rel_resid:
            why["high_residual"] += 1
            continue
        if abs(b) > 0.3 * np.median(v["z"][m]):  # implausible shift -> scale-only fit
            a, b = float(np.median(v["z"][m] / d[m])), 0.0
        depth = (a * v["depth_raw"] + b).astype(np.float32)
        # reject flying pixels at depth discontinuities
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        depth[(np.hypot(gx, gy) / np.maximum(depth, 1e-6)) > 0.6] = 0
        depth[depth < 0] = 0
        v["depth"], v["a"], v["ok"] = depth, a, True
        good.append(v)
        scales.append(a)
    print(f"alignment rejects: {why}")
    if why["negative_scale"] > 0.5 * len(views):
        print("DIAGNOSIS: monocular depth is anti-correlated with SfM depth on most frames. The SfM model is "
              "probably depth-flipped/degenerate (common with global SfM without a focal prior, or with almost no "
              "camera translation). Re-run run_sfm.py with a --focal-factor close to your camera or --mapper incremental.")
    return good, np.array(scales)


def integrate(good, lo, hi, vox, sfm_to_m, trunc_vox, device):
    import numpy as np
    import torch
    dims = np.ceil((hi - lo) / vox).astype(np.int64)
    N = int(np.prod(dims))
    print(f"TSDF grid {tuple(dims)} = {N / 1e6:.1f}M voxels, voxel {vox * sfm_to_m * 100:.1f} cm", flush=True)
    trunc = trunc_vox * vox
    tsdf = torch.zeros(N, dtype=torch.float32, device=device)
    wt = torch.zeros(N, dtype=torch.float32, device=device)
    col = torch.zeros(N, 3, dtype=torch.float16, device=device)
    cwt = torch.zeros(N, dtype=torch.float32, device=device)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=device)
    ny, nz = int(dims[1]), int(dims[2])
    CH = 2_000_000
    t0 = time.time()
    for fi, v in enumerate(good):
        R = torch.tensor(v["R"], dtype=torch.float32, device=device)
        t = torch.tensor(v["t"], dtype=torch.float32, device=device)
        fx, fy, cx, cy = v["K"][0, 0], v["K"][1, 1], v["K"][0, 2], v["K"][1, 2]
        W, H = v["W"], v["H"]
        depth = torch.tensor(v["depth"], device=device)
        rgb = torch.tensor(v["rgb"], device=device).to(torch.float16) / 255.0
        for i0 in range(0, N, CH):
            idx = torch.arange(i0, min(N, i0 + CH), device=device)
            ix = idx // (ny * nz)
            rem = idx - ix * (ny * nz)
            iy = rem // nz
            iz = rem - iy * nz
            pts = lo_t + (torch.stack([ix, iy, iz], 1).float() + 0.5) * vox
            cam = pts @ R.T + t
            z = cam[:, 2]
            zs = z.clamp(min=1e-6)
            u = torch.round(fx * cam[:, 0] / zs + cx).long()
            vv = torch.round(fy * cam[:, 1] / zs + cy).long()
            ins = (z > 1e-6) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            if not ins.any():
                continue
            d = depth[vv.clamp(0, H - 1), u.clamp(0, W - 1)]
            sdf = d - z
            valid = ins & (d > 0) & (sdf > -trunc)
            sel = torch.nonzero(valid).squeeze(1)
            if sel.numel() == 0:
                continue
            gi = idx[sel]
            val = (sdf[sel] / trunc).clamp(-1, 1)
            dm = d[sel] * sfm_to_m
            w = 1.0 / (1.0 + dm * dm)
            wo = wt[gi]
            tsdf[gi] = (tsdf[gi] * wo + val * w) / (wo + w)
            wt[gi] = wo + w
            near = val.abs() < 0.5
            if near.any():
                gn, wn = gi[near], w[near]
                c = rgb[vv[sel][near], u[sel][near]].float()
                co = cwt[gn]
                col[gn] = ((col[gn].float() * co[:, None] + c * wn[:, None]) / (co + wn)[:, None]).half()
                cwt[gn] = co + wn
        if (fi + 1) % 25 == 0 or fi + 1 == len(good):
            print(f"  fused {fi + 1}/{len(good)}  ({time.time() - t0:.0f}s)", flush=True)
    shape = (int(dims[0]), ny, nz)
    return (tsdf.view(shape).cpu().numpy(), wt.view(shape).cpu().numpy(),
            col.view(*shape, 3).cpu().numpy(), cwt.view(shape).cpu().numpy(), dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="colmap/dense")
    ap.add_argument("--out", default="export/room_raw.ply")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    ap.add_argument("--online", action="store_true", help="allow Hugging Face downloads (default: offline cache only)")
    ap.add_argument("--voxel", type=float, default=0.03, help="voxel size in meters")
    ap.add_argument("--trunc-voxels", type=float, default=4.0)
    ap.add_argument("--max-voxels", type=int, default=30_000_000)
    ap.add_argument("--depth-side", type=int, default=800, help="working resolution (longer side, px)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all registered frames")
    ap.add_argument("--min-weight", type=float, default=0.2)
    ap.add_argument("--min-component-frac", type=float, default=0.02)
    ap.add_argument("--min-sparse-pts", type=int, default=40)
    ap.add_argument("--max-rel-resid", type=float, default=0.15)
    a = ap.parse_args()
    if not a.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import torch
    import trimesh
    from scipy.ndimage import map_coordinates
    from skimage import measure

    rec = load_recon(a.work)
    print(f"Registered images in model: {len(rec.images)}")
    views = gather_views(rec, os.path.join(a.work, "images"), a.depth_side, a.max_frames)
    print(f"Views loaded: {len(views)}")

    predict_depths(views, a.model)
    good, scales = align_depths(views, a.min_sparse_pts, a.max_rel_resid)
    if len(good) < 10:
        raise SystemExit(f"Only {len(good)} frames aligned to sparse points. Check SfM quality and capture.")
    a_med = float(np.median(scales))
    cv = float(1.4826 * np.median(np.abs(scales - a_med)) / a_med)
    sfm_to_m = 1.0 / a_med
    print(f"Aligned {len(good)}/{len(views)} frames. SfM->meters = {sfm_to_m:.5f}; scale spread (robust CV) = {cv:.3f}")
    if cv > 0.15:
        print("WARNING: depth scale varies a lot between frames; expect warped geometry. Check lighting/capture.")

    allpts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    centers = np.array([-v["R"].T @ v["t"] for v in good])
    lo = np.minimum(np.percentile(allpts, 1, axis=0), centers.min(0))
    hi = np.maximum(np.percentile(allpts, 99, axis=0), centers.max(0))
    pad = 0.10 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    vox = a.voxel / sfm_to_m
    while np.prod(np.ceil((hi - lo) / vox)) > a.max_voxels:
        vox *= 1.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA; integration will be slow. Lower --max-voxels / --max-frames.")
    tsdf, wt, col, cwt, dims = integrate(good, lo, hi, vox, sfm_to_m, a.trunc_voxels, device)

    mask = wt >= a.min_weight
    if mask.sum() < 1000:
        raise SystemExit("Almost nothing observed in the volume; check alignment output.")
    verts, faces, _, _ = measure.marching_cubes(tsdf, level=0.0, spacing=(vox,) * 3, mask=mask)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    # make faces point toward free space (positive TSDF)
    probe = mesh.vertices + mesh.vertex_normals * (1.5 * vox)
    if map_coordinates(tsdf, (probe / vox).T, order=1, mode="nearest").mean() < 0:
        mesh.invert()
    idx = np.clip(np.round(mesh.vertices / vox).astype(int), 0, dims - 1)
    colors = np.clip(col[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32) * 255, 0, 255).astype(np.uint8)
    colors[cwt[idx[:, 0], idx[:, 1], idx[:, 2]] <= 0] = 128
    mesh.visual.vertex_colors = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], 1)
    mesh.vertices = mesh.vertices + lo + 0.5 * vox  # array space -> SfM world

    parts = mesh.split(only_watertight=False)
    biggest = max(len(p.faces) for p in parts)
    kept = [p for p in parts if len(p.faces) >= a.min_component_frac * biggest]
    mesh = trimesh.util.concatenate(kept)
    print(f"Mesh: {len(mesh.faces)} tris, kept {len(kept)}/{len(parts)} components")

    # metric scale, gravity-aligned Z-up, floor at Z=0
    down = np.mean([v["R"][1, :] for v in good], axis=0)
    up = -down / np.linalg.norm(down)
    T = trimesh.geometry.align_vectors(up, [0, 0, 1])
    mesh.apply_scale(sfm_to_m)
    mesh.apply_transform(T)
    floor = float(np.percentile(mesh.vertices[:, 2], 1))
    mesh.apply_translation([0, 0, -floor])
    c0 = centers[0] * sfm_to_m
    spawn = (T @ np.append(c0, 1.0))[:3] - np.array([0, 0, floor])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mesh.export(a.out)
    import pycolmap
    meta = dict(params=vars(a), versions=dict(numpy=np.__version__, torch=torch.__version__,
                trimesh=trimesh.__version__, pycolmap=pycolmap.__version__),
                frames_registered=len(rec.images), frames_aligned=len(good), sfm_to_m=sfm_to_m, scale_cv=cv,
                voxel_m=vox * sfm_to_m, triangles=int(len(mesh.faces)), bounds_m=mesh.bounds.tolist(),
                spawn_xyz_zup=spawn.tolist(), floor_shift_m=-floor)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {a.out} ({len(mesh.faces)} tris). Bounds (m): {np.round(mesh.bounds, 2).tolist()}")
    print("Check by eye: room size plausible? floor flat at Z=0? door height right? Then open the PLY in Blender.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
