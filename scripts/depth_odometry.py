"""COLMAP-free reconstruction: metric depth + GPU feature matching + PnP odometry + pose graph + TSDF mesh.

Pipeline (all on the GPU where it matters):
  1. Depth-Anything-V2-Metric-Indoor (cached, offline) -> metric depth per frame.
  2. SIFT keypoints (OpenCV), matched on the GPU (mutual nearest neighbour + ratio test).
  3. For frame pairs (i, i+1..i+k): lift frame-i keypoints with metric depth -> PnP+RANSAC -> relative pose.
     The depth ratio at matched points also gives the relative depth scale between frames.
  4. Solve per-frame depth scales (robust least squares), then a pose graph over all pair measurements
     (robust loss, outlier edges pruned). Loop-closure edges are added between frames that revisit a place.
  5. Fuse depth into a TSDF on the GPU, mesh, align (meters, Z-up, floor at 0). See fuse.py.

Usage (project root):
    python -I scripts/depth_odometry.py --frames frames_fast --out export/room_odo.ply

ASSUMPTIONS to check: focal = --focal-factor * longer side (0.69 ~ phone main camera) and no lens distortion;
phone held upright; metric depth is roughly right in absolute terms (the model is trained for indoor scenes).
Known limits: drift on long walks without revisits; monocular depth is weak on mirrors/glass/blank walls.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_frames(frames_dir, side, stride, max_frames):
    import cv2
    import numpy as np
    names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png")))[::stride]
    if max_frames and len(names) > max_frames:
        names = [names[i] for i in np.linspace(0, len(names) - 1, max_frames).round().astype(int)]
    views = []
    for n in names:
        bgr = cv2.imread(os.path.join(frames_dir, n))
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]
        s = min(1.0, side / max(h0, w0))
        W, H = int(round(w0 * s)), int(round(h0 * s))
        if (W, H) != (w0, h0):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        views.append(dict(name=n, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), W=W, H=H))
    return views


def extract_features(views, nfeat, device):
    import cv2
    import numpy as np
    import torch
    sift = cv2.SIFT_create(nfeatures=nfeat)
    for v in views:
        kp, des = sift.detectAndCompute(cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2GRAY), None)
        v["kp"] = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2)
        v["des"] = torch.from_numpy(des).to(device) if des is not None and len(des) else None


def match(di, dj, ratio=0.8):
    import torch
    if di is None or dj is None or len(di) < 8 or len(dj) < 8:
        return None
    D = torch.cdist(di, dj)
    v, idx = D.topk(2, dim=1, largest=False)
    ok = v[:, 0] < ratio * v[:, 1]
    back = D.argmin(dim=0)
    ok &= back[idx[:, 0]] == torch.arange(len(di), device=di.device)
    qi = torch.nonzero(ok).squeeze(1)
    return qi.cpu().numpy(), idx[qi, 0].cpu().numpy()


def backproject(p, d, K):
    import numpy as np
    return np.stack([(p[:, 0] - K[0, 2]) / K[0, 0] * d, (p[:, 1] - K[1, 2]) / K[1, 1] * d, d], 1)


def estimate_pair(vi, vj, K, a):
    """Relative pose cam_j_from_cam_i (translation in frame-i depth units) and depth ratio s_j/s_i."""
    import cv2
    import numpy as np
    m = match(vi["des"], vj["des"])
    if m is None or len(m[0]) < a.min_inliers:
        return None
    qi, qj = m
    pi, pj = vi["kp"][qi].astype(np.float64), vj["kp"][qj].astype(np.float64)
    xi = np.clip(np.round(pi[:, 0]).astype(int), 0, vi["W"] - 1)
    yi = np.clip(np.round(pi[:, 1]).astype(int), 0, vi["H"] - 1)
    d = vi["depth_raw"][yi, xi].astype(np.float64)
    ok = (d > 0.3) & (d < a.max_depth)
    if ok.sum() < a.min_inliers:
        return None
    pi, pj, d = pi[ok], pj[ok], d[ok]
    X = backproject(pi, d, K)
    good, rvec, tvec, inl = cv2.solvePnPRansac(X, pj, K, None, iterationsCount=500, reprojectionError=3.0,
                                               confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    if not good or inl is None or len(inl) < a.min_inliers:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X[inl], pj[inl], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    if np.linalg.norm(t) > a.max_step:
        return None
    xj = np.clip(np.round(pj[inl, 0]).astype(int), 0, vj["W"] - 1)
    yj = np.clip(np.round(pj[inl, 1]).astype(int), 0, vj["H"] - 1)
    dj = vj["depth_raw"][yj, xj].astype(np.float64)
    zp = (X[inl] @ R.T + t)[:, 2]
    m2 = (dj > 0.3) & (zp > 0.1)
    if m2.sum() < 10:
        return None
    rho = float(np.median(dj[m2] / zp[m2]))
    if not 0.5 < rho < 2.0:
        return None
    return dict(R=R, t=t, inl=int(len(inl)), rho=rho)


def set_intrinsics(views, factor):
    import numpy as np
    for v in views:
        f = factor * max(v["W"], v["H"])
        v["K"] = np.array([[f, 0, v["W"] / 2], [0, f, v["H"] / 2], [0, 0, 1]], dtype=np.float64)


def calibrate_focal(views, a, gap, n_pairs=14):
    """Self-calibrate focal length (as a fraction of the longer side) from the footage itself.
    Metric depth does not depend on the focal length, so a wrong focal makes the depth-lifted 3D points
    inconsistent with the 2D matches in the next frame: PnP inlier counts peak near the true focal."""
    import cv2
    import numpy as np
    n = len(views)
    gap = max(1, min(gap, n - 1))
    idx = np.linspace(0, n - 1 - gap, min(n_pairs, n - gap)).round().astype(int)
    cached = []
    for i in dict.fromkeys(idx.tolist()):
        j = i + gap
        m = match(views[i]["des"], views[j]["des"])
        if m is None or len(m[0]) < 40:
            continue
        pi = views[i]["kp"][m[0]].astype(np.float64)
        pj = views[j]["kp"][m[1]].astype(np.float64)
        xi = np.clip(np.round(pi[:, 0]).astype(int), 0, views[i]["W"] - 1)
        yi = np.clip(np.round(pi[:, 1]).astype(int), 0, views[i]["H"] - 1)
        d = views[i]["depth_raw"][yi, xi].astype(np.float64)
        ok = (d > 0.3) & (d < a.max_depth)
        if ok.sum() >= 40:
            cached.append((pi[ok], pj[ok], d[ok]))
    if len(cached) < 3:
        return None, {}
    W, H = views[0]["W"], views[0]["H"]

    def score(fac):
        f = fac * max(W, H)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        tot = 0
        for pi, pj, d in cached:
            ok, _, _, inl = cv2.solvePnPRansac(backproject(pi, d, K), pj, K, None, iterationsCount=300,
                                               reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)
            tot += len(inl) if ok and inl is not None else 0
        return tot

    scores = {fac: score(fac) for fac in (0.45, 0.55, 0.65, 0.75, 0.9, 1.1, 1.4)}
    best = max(scores, key=scores.get)
    for fac in np.arange(best - 0.1, best + 0.1001, 0.025):
        fac = round(float(fac), 3)
        if fac > 0.2 and fac not in scores:
            scores[fac] = score(fac)
    best = max(scores, key=scores.get)
    if scores[best] < 1.15 * np.median(list(scores.values())):  # flat curve -> not informative
        return None, scores
    return best, scores


def components(n, edges):
    adj = {}
    for e in edges:
        adj.setdefault(e["i"], set()).add(e["j"])
        adj.setdefault(e["j"], set()).add(e["i"])
    seen, comps = set(), []
    for s in range(n):
        if s in seen or s not in adj:
            continue
        comp, stack = set(), [s]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def solve_scales(n, edges, root):
    import numpy as np
    A = np.zeros((len(edges) + 1, n))
    b = np.zeros(len(edges) + 1)
    base_w = np.sqrt(np.minimum([e["inl"] for e in edges], 200) / 200.0)
    w = base_w.copy()
    for _ in range(4):  # IRLS (Huber-like)
        A[:] = 0
        for k, e in enumerate(edges):
            A[k, e["j"]], A[k, e["i"]], b[k] = w[k], -w[k], w[k] * np.log(e["rho"])
        A[-1, root] = 1e3
        u = np.linalg.lstsq(A, b, rcond=None)[0]
        r = np.array([np.log(e["rho"]) - (u[e["j"]] - u[e["i"]]) for e in edges])
        s = 1.4826 * np.median(np.abs(r)) + 1e-3
        w = base_w / np.maximum(1.0, np.abs(r) / (1.5 * s))
    return np.exp(u)


def init_poses(nodes, edges):
    import numpy as np
    adj = {}
    for e in edges:
        adj.setdefault(e["i"], []).append((e["j"], e["R"], e["tc"]))
        adj.setdefault(e["j"], []).append((e["i"], e["R"].T, -e["R"].T @ e["tc"]))
    pose = {nodes[0]: (np.eye(3), np.zeros(3))}
    stack = [nodes[0]]
    while stack:
        c = stack.pop()
        Rc, tc = pose[c]
        for nb, Rr, tr in adj.get(c, []):
            if nb not in pose:
                pose[nb] = (Rr @ Rc, Rr @ tc + tr)
                stack.append(nb)
    return pose


def pose_graph(nodes, edges, pose0, sig_r, sig_t):
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation as Rot
    n = len(nodes)
    loc = {g: k for k, g in enumerate(nodes)}
    I = np.array([loc[e["i"]] for e in edges])
    J = np.array([loc[e["j"]] for e in edges])
    Rm = np.stack([e["R"] for e in edges])
    tm = np.stack([e["tc"] for e in edges])
    w = np.array([min(1.5, max(0.3, e["inl"] / 100.0)) for e in edges])
    R0 = np.stack([pose0[g][0] for g in nodes])
    t0 = np.stack([pose0[g][1] for g in nodes])
    x0 = np.concatenate([Rot.from_matrix(R0[1:]).as_rotvec(), t0[1:]], 1).ravel()

    def unpack(x):
        P = x.reshape(n - 1, 6)
        R = np.concatenate([np.eye(3)[None], Rot.from_rotvec(P[:, :3]).as_matrix()])
        t = np.concatenate([np.zeros((1, 3)), P[:, 3:]])
        return R, t

    def resid(x):
        R, t = unpack(x)
        Rji = R[J] @ np.transpose(R[I], (0, 2, 1))
        tji = t[J] - np.einsum("nij,nj->ni", Rji, t[I])
        rr = Rot.from_matrix(np.transpose(Rm, (0, 2, 1)) @ Rji).as_rotvec() / sig_r
        rt = (tji - tm) / sig_t
        return (np.concatenate([rr, rt], 1) * w[:, None]).ravel()

    sp = lil_matrix((6 * len(edges), 6 * (n - 1)), dtype=int)
    for k, (i, j) in enumerate(zip(I, J)):
        for g in (i, j):
            if g > 0:
                sp[6 * k:6 * k + 6, 6 * (g - 1):6 * (g - 1) + 6] = 1
    sol = least_squares(resid, x0, jac_sparsity=sp, loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=100)
    R, t = unpack(sol.x)
    rms = np.sqrt(np.mean((resid(sol.x).reshape(len(edges), 6) / w[:, None]) ** 2, axis=1))
    return R, t, rms


def solve(n, edges, a):
    """Scales + pose graph over `edges`; prunes outlier edges once. Returns dict(nodes, R, t, s, edges, rms)."""
    import numpy as np
    comps = components(n, edges)
    if not comps:
        raise SystemExit("No frame pairs could be matched. Capture needs more overlap/texture.")
    big = max(comps, key=len)
    nodes = sorted(big)
    for attempt in range(2):
        edges = [e for e in edges if e["i"] in big and e["j"] in big]
        s = solve_scales(n, edges, nodes[0])
        for e in edges:
            e["tc"] = e["t"] / s[e["i"]]
        pose0 = init_poses(nodes, edges)
        R, t, rms = pose_graph(nodes, edges, pose0, a.sigma_rot, a.sigma_trans)
        if attempt == 1:
            break
        keep = [e for e, r in zip(edges, rms) if r < a.prune_rms]
        if len(keep) == len(edges):
            break
        print(f"  pruned {len(edges) - len(keep)} outlier edges (rms > {a.prune_rms})")
        comps = components(n, keep)
        big = max(comps, key=len)
        nodes = sorted(big)
        edges = keep
    return dict(nodes=nodes, R=R, t=t, s=s, edges=edges, rms=rms)


def consistency_filter(good, k, tol, min_agree, device):
    """Keep a depth pixel only if >= min_agree neighbouring frames (within +-k) see the same depth (rel. tol).
    Removes monocular-depth outliers and pose-inconsistent pixels that otherwise smear the TSDF."""
    import numpy as np
    import torch
    n = len(good)
    D = [torch.tensor(g["depth"], device=device) for g in good]
    Rt = [(torch.tensor(g["R"], dtype=torch.float32, device=device),
           torch.tensor(g["t"], dtype=torch.float32, device=device)) for g in good]
    K, H, W = good[0]["K"], good[0]["H"], good[0]["W"]
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    vv, uu = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    rays = torch.stack([(uu - cx) / f, (vv - cy) / f, torch.ones_like(uu, dtype=torch.float32)], -1).float()
    kept = []
    for i in range(n):
        Ri, ti = Rt[i]
        Xw = (rays * D[i][..., None] - ti) @ Ri          # world = R^T (X - t)
        agree = torch.zeros(H, W, device=device)
        for j in range(max(0, i - k), min(n, i + k + 1)):
            if j == i:
                continue
            Rj, tj = Rt[j]
            Xj = Xw @ Rj.T + tj
            z = Xj[..., 2]
            u = torch.round(f * Xj[..., 0] / z.clamp(min=1e-6) + cx).long()
            v = torch.round(f * Xj[..., 1] / z.clamp(min=1e-6) + cy).long()
            ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (D[i] > 0)
            dj = D[j][v.clamp(0, H - 1), u.clamp(0, W - 1)]
            ok &= dj > 0
            agree += (ok & ((dj - z).abs() < tol * z)).float()
        before = int((D[i] > 0).sum())
        keep = agree >= min_agree
        good[i]["depth"] = (D[i] * keep).cpu().numpy()
        kept.append(int(keep.sum()) / max(1, before))
        good[i]["kept_frac"] = kept[-1]  # callers can drop frames that agree with none of their neighbours
    print(f"consistency filter: kept {100 * np.mean(kept):.0f}% of depth pixels on average "
          f"(min {100 * np.min(kept):.0f}%, max {100 * np.max(kept):.0f}%)", flush=True)
    return float(np.mean(kept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--out", default="export/room_odo.ply")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--focal-factor", default="auto",
                    help="'auto' = self-calibrate from the footage (falls back to 0.69 if inconclusive), "
                         "or a number = focal length / longer image side")
    ap.add_argument("--side", type=int, default=800, help="working resolution (longer side, px)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--neighbors", type=int, default=0, help="pair each frame with this many following frames "
                                                              "(0 = derive from --neighbor-seconds and the kept fps)")
    ap.add_argument("--neighbor-seconds", type=float, default=0.6)
    ap.add_argument("--loop-gap-seconds", type=float, default=3.0,
                    help="loop closures only between frames at least this far apart in time")
    ap.add_argument("--nfeat", type=int, default=3000)
    ap.add_argument("--min-inliers", type=int, default=40)
    ap.add_argument("--max-depth", type=float, default=8.0)
    ap.add_argument("--max-step", type=float, default=3.0, help="max plausible translation between paired frames (m)")
    ap.add_argument("--sigma-rot", type=float, default=0.01)
    ap.add_argument("--sigma-trans", type=float, default=0.03)
    ap.add_argument("--prune-rms", type=float, default=6.0)
    ap.add_argument("--loop-gap", type=int, default=0, help="frames (0 = derive from --loop-gap-seconds)")
    ap.add_argument("--loop-dist", type=float, default=2.0)
    ap.add_argument("--loops-per-frame", type=int, default=3)
    ap.add_argument("--loop-min-inliers", type=int, default=60)
    ap.add_argument("--edge-grad", type=float, default=0.3, help="drop depth pixels with |grad|/depth above this")
    ap.add_argument("--consist-k", type=int, default=3, help="neighbour frames each side for depth consistency")
    ap.add_argument("--consist-tol", type=float, default=0.08, help="relative depth tolerance")
    ap.add_argument("--consist-min", type=int, default=2, help="min agreeing neighbours to keep a pixel")
    ap.add_argument("--no-consist", action="store_true", help="disable the multi-view depth consistency filter")
    ap.add_argument("--scale", type=float, default=1.0, help="uniform scale multiplier on the result")
    ap.add_argument("--ceiling-height", type=float, default=0.0,
                    help="REAL floor-to-ceiling height in meters; the mesh is rescaled so its detected "
                         "floor-to-ceiling matches. Monocular depth cannot know absolute size, so give a measured value.")
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
    from dense_tsdf import predict_depths
    from fuse import fuse_mesh_export

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T0 = time.time()
    views = load_frames(a.frames, a.side, a.stride, a.max_frames)
    n = len(views)
    if n < 10:
        raise SystemExit(f"Only {n} frames in {a.frames}.")
    print(f"{n} frames loaded ({views[0]['W']}x{views[0]['H']}), device={device}", flush=True)
    import json
    cv2.setRNGSeed(0)  # reproducible RANSAC
    auto_focal = str(a.focal_factor).lower() == "auto"
    fac_used = 0.69 if auto_focal else float(a.focal_factor)  # provisional until calibrated
    focal_scores = {}
    set_intrinsics(views, fac_used)
    K = views[0]["K"]
    kept_fps = None
    mp = os.path.join(a.frames, "meta.json")
    if os.path.isfile(mp):
        with open(mp) as fh:
            kept_fps = json.load(fh).get("kept_fps")
        if kept_fps:
            kept_fps /= max(1, a.stride)
    if a.neighbors <= 0:
        a.neighbors = int(np.clip(round(a.neighbor_seconds * kept_fps), 2, 8)) if kept_fps else 3
    if a.loop_gap <= 0:
        a.loop_gap = max(5, int(round(a.loop_gap_seconds * kept_fps))) if kept_fps else 15
    print(f"kept fps: {kept_fps}; neighbors={a.neighbors}; loop gap={a.loop_gap} frames", flush=True)

    predict_depths(views, a.model)
    print(f"[{time.time() - T0:.0f}s] depth done", flush=True)
    extract_features(views, a.nfeat, device)
    print(f"[{time.time() - T0:.0f}s] features: median {int(np.median([len(v['kp']) for v in views]))}/frame", flush=True)
    if auto_focal:
        best, focal_scores = calibrate_focal(views, a, a.neighbors)
        if best is None:
            print("focal self-calibration inconclusive; using default 0.69 x longer side", flush=True)
        else:
            fac_used = best
            set_intrinsics(views, fac_used)
            K = views[0]["K"]
            print(f"focal self-calibrated: {fac_used:.3f} x longer side "
                  f"(inliers by factor: { {k: focal_scores[k] for k in sorted(focal_scores)} })", flush=True)

    edges, tried = [], 0
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + a.neighbors)):
            tried += 1
            r = estimate_pair(views[i], views[j], K, a)
            if r:
                edges.append(dict(i=i, j=j, **r))
    print(f"[{time.time() - T0:.0f}s] neighbor pairs: {len(edges)}/{tried} succeeded", flush=True)

    sol = solve(n, edges, a)
    print(f"[{time.time() - T0:.0f}s] stage 1: {len(sol['nodes'])}/{n} frames connected, median edge rms {np.median(sol['rms']):.2f}", flush=True)

    # loop closures between frames that end up close in space but far apart in time
    have = {(e["i"], e["j"]) for e in sol["edges"]}
    nodes = sol["nodes"]
    C = np.array([-sol["R"][k].T @ sol["t"][k] for k in range(len(nodes))])
    Zax = np.array([sol["R"][k][2] for k in range(len(nodes))])
    new = []
    for ka, ia in enumerate(nodes):
        d = np.linalg.norm(C - C[ka], axis=1)
        cand = [kb for kb in np.argsort(d) if nodes[kb] - ia > a.loop_gap and d[kb] < a.loop_dist
                and Zax[ka] @ Zax[kb] > 0.5 and (ia, nodes[kb]) not in have][:a.loops_per_frame]
        for kb in cand:
            r = estimate_pair(views[ia], views[nodes[kb]], K, a)
            if r and r["inl"] >= a.loop_min_inliers:
                new.append(dict(i=ia, j=nodes[kb], **r))
    print(f"[{time.time() - T0:.0f}s] loop closures found: {len(new)}", flush=True)
    if new:
        sol = solve(n, sol["edges"] + new, a)
        print(f"[{time.time() - T0:.0f}s] stage 2: {len(sol['nodes'])}/{n} frames, median edge rms {np.median(sol['rms']):.2f}", flush=True)

    nodes, Rs, ts, s = sol["nodes"], sol["R"], sol["t"], sol["s"]
    if len(nodes) < 10:
        raise SystemExit("Fewer than 10 frames connected. Recapture with more overlap/texture.")
    good = []
    for k, g in enumerate(nodes):
        v = views[g]
        depth = (v["depth_raw"] / s[g]).astype(np.float32)
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        depth[(np.hypot(gx, gy) / np.maximum(depth, 1e-6)) > a.edge_grad] = 0
        depth[(depth < 0.3) | (depth > a.max_depth)] = 0
        good.append(dict(R=Rs[k], t=ts[k], K=v["K"], W=v["W"], H=v["H"], rgb=v["rgb"], depth=depth, name=v["name"]))
    if not a.no_consist:
        consistency_filter(good, a.consist_k, a.consist_tol, a.consist_min, device)
    pts = []
    for k, gd in enumerate(good):
        sub = gd["depth"][::16, ::16]
        yy, xx = np.nonzero(sub > 0)
        z = sub[yy, xx].astype(np.float64)
        uv = np.stack([xx * 16.0, yy * 16.0], 1)
        Xc = backproject(uv, z, gd["K"])
        pts.append((Xc - ts[k]) @ Rs[k])  # world = R^T (Xc - t)
    pts = np.concatenate(pts)
    steps = np.linalg.norm(np.diff(np.array([-Rs[k].T @ ts[k] for k in range(len(nodes))]), axis=0), axis=1)
    print(f"trajectory length {steps.sum():.1f} m over {len(nodes)} frames; depth scale spread (std of log s) "
          f"{np.std(np.log(s[nodes])):.3f}", flush=True)

    fuse_mesh_export(good, pts, a.scale, a, dict(method="depth_odometry", frames_total=n, frames_connected=len(nodes),
                                            edges=len(sol["edges"]), loops=len(new),
                                            median_edge_rms=float(np.median(sol["rms"])),
                                            focal_factor=fac_used, focal_scores=focal_scores))
    print(f"[{time.time() - T0:.0f}s] done. Check: room size, flat floor at Z=0, straight walls. "
          "Then open the PLY in Blender.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
