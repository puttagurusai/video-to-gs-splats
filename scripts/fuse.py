"""Shared back end: TSDF fusion (GPU) -> marching cubes -> clean -> metric/Z-up/floor alignment -> PLY + JSON.

Used by depth_odometry.py. `good` is a list of dicts with keys R, t (cam_from_world), K, W, H, rgb, depth
(depth in the same units as the poses). `allpts` are world points used only to size the volume.
"""
import json
import os
import sys


def _refine_up(mesh, up0, max_deg=20.0, iters=6):
    """Refine gravity from surface normals: floor/ceiling vertices have normals along +/-up."""
    import numpy as np
    n = np.asarray(mesh.vertex_normals)
    up = up0 / np.linalg.norm(up0)
    c = np.cos(np.radians(max_deg))
    for _ in range(iters):
        d = n @ up
        sel = np.abs(d) > c
        if sel.sum() < 500:
            break
        m = (n[sel] * np.sign(d[sel])[:, None]).mean(0)
        up = m / np.linalg.norm(m)
    return up, float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1))))


def _floor_ceiling(mesh, bin_m=0.05, frac=0.3):
    """Floor = lowest strong peak among up-facing vertices; ceiling = highest strong peak among down-facing."""
    import numpy as np
    n = np.asarray(mesh.vertex_normals)
    z = mesh.vertices[:, 2]

    def peaks(sel):
        if sel.sum() < 500:
            return None, None
        h, e = np.histogram(z[sel], bins=np.arange(z[sel].min(), z[sel].max() + bin_m, bin_m))
        g = np.nonzero(h >= frac * h.max())[0]
        return (e[g[0]] + e[g[0] + 1]) / 2, (e[g[-1]] + e[g[-1] + 1]) / 2

    floor, _ = peaks(n[:, 2] > 0.85)
    _, ceil = peaks(n[:, 2] < -0.85)
    return floor, ceil


def _choose_up(mesh, up0, cam_centers, cos_deg=15.0):
    """Gravity prior = mean camera 'down' (good for upright handheld room scans). If almost no surface is
    perpendicular to it (e.g. a steep top-down shot of a table), use the dominant surface normal instead,
    signed so the cameras sit on its positive side."""
    import numpy as np
    n = np.asarray(mesh.vertex_normals)
    c = np.cos(np.radians(cos_deg))
    if np.mean(np.abs(n @ up0) > c) >= 0.10:
        return up0, "camera orientation"
    rng = np.random.default_rng(0)
    ns = n[rng.choice(len(n), min(len(n), 40000), replace=False)].astype(np.float32)
    k = np.arange(500) + 0.5
    phi = np.arccos(1 - 2 * k / 500)
    theta = np.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1).astype(np.float32)
    share = (np.abs(ns @ dirs.T) > c).mean(0)
    best = dirs[int(np.argmax(share))].astype(np.float64)
    if share.max() < 0.25:
        return up0, "camera orientation (weak surface evidence)"
    V = np.asarray(mesh.vertices)
    side = cam_centers.mean(0) @ best - np.median(V @ best)
    return (best if side >= 0 else -best), "dominant surface"


def fuse_mesh_export(good, allpts, sfm_to_m, a, extra_meta=None):
    import time
    import numpy as np
    t_start = time.time()
    import torch
    import trimesh
    from scipy.ndimage import map_coordinates
    from skimage import measure

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dense_tsdf import integrate

    centers = np.array([-v["R"].T @ v["t"] for v in good])
    lo = np.minimum(np.percentile(allpts, 1, axis=0), centers.min(0))
    hi = np.maximum(np.percentile(allpts, 99, axis=0), centers.max(0))
    pad = 0.10 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ext_m = (hi - lo) * sfm_to_m
    if str(a.voxel).lower() == "auto":  # ~220 cells along the longest axis, clamped to a sensible range
        vox_m = float(np.clip(ext_m.max() / 220.0, 0.003, 0.04))
    else:
        vox_m = float(a.voxel)
    vox = vox_m / sfm_to_m
    while np.prod(np.ceil((hi - lo) / vox)) > a.max_voxels:
        vox *= 1.1
    print(f"scene extent {np.round(ext_m, 2).tolist()} m -> voxel {vox * sfm_to_m * 100:.2f} cm "
          f"({'auto' if str(a.voxel).lower() == 'auto' else 'fixed'})", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA; integration will be slow. Lower --max-voxels / --max-frames.")
    tsdf, wt, col, cwt, dims = integrate(good, lo, hi, vox, sfm_to_m, a.trunc_voxels, device)

    print(f"  [fuse] TSDF integration done at {time.time() - t_start:.0f}s", flush=True)
    mask = wt >= a.min_weight
    if mask.sum() < 1000:
        raise SystemExit("Almost nothing observed in the volume; check poses/depth.")
    verts, faces, _, _ = measure.marching_cubes(tsdf, level=0.0, spacing=(vox,) * 3, mask=mask)
    print(f"  [fuse] marching cubes done at {time.time() - t_start:.0f}s ({len(faces)} tris)", flush=True)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    probe = mesh.vertices + mesh.vertex_normals * (1.5 * vox)  # faces should point toward free space (TSDF > 0)
    if map_coordinates(tsdf, (probe / vox).T, order=1, mode="nearest").mean() < 0:
        mesh.invert()
    idx = np.clip(np.round(mesh.vertices / vox).astype(int), 0, dims - 1)
    colors = np.clip(col[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32) * 255, 0, 255).astype(np.uint8)
    colored = cwt[idx[:, 0], idx[:, 1], idx[:, 2]] > 0
    colors[~colored] = 128
    mesh.visual.vertex_colors = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], 1)
    if getattr(a, "drop_uncolored", True):  # surfaces no pixel ever observed near the surface are extrapolation
        ok_f = colored[mesh.faces].all(axis=1)
        print(f"  dropped {int((~ok_f).sum())} of {len(ok_f)} faces with no colour support", flush=True)
        mesh.update_faces(ok_f)
        mesh.remove_unreferenced_vertices()
    mesh.vertices = mesh.vertices + lo + 0.5 * vox  # array space -> world

    # drop small floating fragments with one sparse-graph pass (mesh.split() builds an object per component and
    # takes minutes on noisy meshes with thousands of components)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
    fa = mesh.face_adjacency
    nF = len(mesh.faces)
    graph = coo_matrix((np.ones(len(fa), np.int8), (fa[:, 0], fa[:, 1])), shape=(nF, nF))
    n_comp, lab = _cc(graph, directed=False)
    sizes = np.bincount(lab, minlength=n_comp)
    biggest = int(sizes.max())
    big_enough = sizes >= a.min_component_frac * biggest
    comp_info = dict(components_total=int(n_comp), largest_component_frac=float(biggest / nF))
    mesh.update_faces(big_enough[lab])
    mesh.remove_unreferenced_vertices()
    print(f"Mesh: {len(mesh.faces)} tris, kept {int(big_enough.sum())}/{n_comp} components "
          f"(largest holds {100 * comp_info['largest_component_frac']:.0f}% of faces) at {time.time() - t_start:.0f}s", flush=True)

    # gravity: camera 'down' vectors as a first guess, refined from surface normals; then floor/ceiling levels
    down = np.mean([v["R"][1, :] for v in good], axis=0)
    up0 = -down / np.linalg.norm(down)
    mesh.apply_scale(sfm_to_m)
    up0, up_src = _choose_up(mesh, up0, centers * sfm_to_m)
    up, tilt_fix = _refine_up(mesh, up0)
    T = trimesh.geometry.align_vectors(up, [0, 0, 1])
    mesh.apply_transform(T)
    floor, ceil = _floor_ceiling(mesh)
    if floor is None:
        floor = float(np.percentile(mesh.vertices[:, 2], 1))
    est_height = float(ceil - floor) if ceil is not None else None
    if est_height is not None and est_height < 1.0:  # no real floor/ceiling pair (e.g. table-top or object scene)
        est_height = None
    rescale = 1.0
    ref_h = float(getattr(a, "ceiling_height", 0) or 0)
    if ref_h > 0 and est_height and est_height > 0.5:  # user-supplied real floor-to-ceiling height
        rescale = ref_h / est_height
        mesh.apply_scale(rescale)
        floor *= rescale
    mesh.apply_translation([0, 0, -floor])
    Tr = np.eye(4)
    Tr[2, 3] = -floor
    world_to_mesh = Tr @ np.diag([rescale] * 3 + [1.0]) @ T @ np.diag([sfm_to_m] * 3 + [1.0])  # SfM/DA3 world -> final mesh frame
    c0 = centers[0] * sfm_to_m
    spawn = (T @ np.append(c0, 1.0))[:3] * rescale - np.array([0, 0, floor])
    level_info = dict(up_tilt_correction_deg=tilt_fix, up_source=up_src, est_floor_to_ceiling_m_before_rescale=est_height,
                      rescale_applied=rescale, **comp_info)
    level_info["world_to_mesh"] = world_to_mesh.tolist()
    mesh.metadata["quality"] = comp_info
    mesh.metadata["world_to_mesh"] = world_to_mesh
    print(f"up from {up_src}, refined by {tilt_fix:.1f} deg; estimated floor-to-ceiling "
          f"{'none (no ceiling found)' if est_height is None else round(est_height, 2)} m"
          + (f"; rescaled x{rescale:.3f} to reference {ref_h} m" if rescale != 1.0 else " (no reference height given)"))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mesh.export(a.out)
    meta = dict(params={k: v for k, v in vars(a).items()},
                versions=dict(numpy=np.__version__, torch=torch.__version__, trimesh=trimesh.__version__),
                frames_used=len(good), voxel_m=vox * sfm_to_m, triangles=int(len(mesh.faces)),
                bounds_m=mesh.bounds.tolist(), spawn_xyz_zup=spawn.tolist(), floor_shift_m=-floor)
    meta.update(extra_meta or {})
    meta.update(level_info)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Wrote {a.out} ({len(mesh.faces)} tris). Bounds (m): {np.round(mesh.bounds, 2).tolist()}")
    return mesh
