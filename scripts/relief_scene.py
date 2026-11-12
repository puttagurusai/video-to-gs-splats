"""Multi-frame "relief" scene: one textured depth mesh per keyframe, placed with the DA3 camera poses.

Each keyframe's own depth map becomes a mesh textured with that same frame at full resolution, so every piece is
self-consistent (this is the single-image depth-to-mesh idea, extended to many frames). Nothing is fused or averaged,
which avoids the layered/blurred surfaces of TSDF fusion. Overlaps are resolved by ownership: a triangle is kept in a
keyframe only if no other keyframe that sees the same surface has a clearly better view (frontal and close).

Run with the DA3 venv python (needs torch):
  tools\\venv_da3\\Scripts\\python.exe -I scripts\\relief_scene.py --cams export\\room_t_cams.npz --out-dir export\\room_relief

Writes <out-dir>/relief.obj (+ relief.mtl and kf_*.jpg textures; Z up, meters) and <out-dir>/views.json (held-out
camera poses for scripts/render_views.py + scripts/eval_views.py).
Limits: pieces do not match perfectly where they meet (depth differs a little between frames); off-path views show
holes behind objects, as any single-photo relief does.
"""
import argparse
import json
import math
import os
import shutil
import time

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--keyframes", type=int, default=24)
    ap.add_argument("--heldout", type=int, default=6)
    ap.add_argument("--stride", type=int, default=3, help="depth pixels per mesh vertex (3 = coarse, 1 = full)")
    ap.add_argument("--edge-tol", type=float, default=0.06, help="cut triangles whose vertex depths differ by more than this (relative)")
    ap.add_argument("--min-cos", type=float, default=0.15, help="drop triangles seen at a grazing angle (stretched)")
    ap.add_argument("--slack", type=float, default=0.85, help="keep a triangle if its score >= slack * best other keyframe (1 = strict, no overlap)")
    ap.add_argument("--depth-tol", type=float, default=0.08)
    a = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = np.load(a.cams)
    names = [str(x) for x in B["names"]]
    R, T = B["R"].astype(np.float64), B["t"].astype(np.float64)
    K = B["K"].astype(np.float64)
    H, W = [int(x) for x in B["hw"]]
    depth, M = B["depth"], B["world_to_mesh"].astype(np.float64)
    fdir = a.frames or str(B["frames_dir"])
    n = len(names)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    M3, mt = M[:3, :3], M[:3, 3]
    s = float(np.cbrt(abs(np.linalg.det(M3))))
    Rot = M3 / s
    C_f = (-np.einsum("nji,nj->ni", R, T)) @ M3.T + mt              # camera centres in the final (mesh) frame

    # keyframes: equal steps along the path (translation + rotation), so pans and walks are both covered
    dC = np.linalg.norm(np.diff(C_f, axis=0), axis=1)
    ang = np.arccos(np.clip((np.einsum("nij,nij->n", R[:-1], R[1:]) - 1) / 2, -1, 1))
    med_d = float(np.median(depth[depth > 0][::997])) * s
    cum = np.concatenate([[0], np.cumsum(dC + 0.5 * med_d * ang)])
    targets = np.linspace(0, cum[-1], min(a.keyframes, n))
    key = sorted(set(np.clip(np.searchsorted(cum, targets), 0, n - 1).tolist()))
    rest = [k for k in range(n) if k not in key]
    ho = sorted(set(rest[i] for i in np.linspace(0, len(rest) - 1, min(a.heldout, len(rest))).round().astype(int))) if rest else []
    print(f"{n} frames -> {len(key)} keyframes {key[:6]}..., held-out for evaluation: {ho}", flush=True)

    ys, xs = torch.arange(0, H, a.stride), torch.arange(0, W, a.stride)
    h, w = len(ys), len(xs)
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    uf, vf = uu.reshape(-1).float() + 0.5, vv.reshape(-1).float() + 0.5
    idx = torch.arange(h * w).view(h, w)
    p00, p01, p10, p11 = idx[:-1, :-1].reshape(-1), idx[:-1, 1:].reshape(-1), idx[1:, :-1].reshape(-1), idx[1:, 1:].reshape(-1)
    tri = torch.cat([torch.stack([p00, p10, p01], 1), torch.stack([p01, p10, p11], 1)], 0).to(dev)
    uf, vf = uf.to(dev), vf.to(dev)

    tR = {i: torch.tensor(R[i], dtype=torch.float32, device=dev) for i in key}
    tT = {i: torch.tensor(T[i], dtype=torch.float32, device=dev) for i in key}
    tC = {i: torch.tensor(C_f[i], dtype=torch.float32, device=dev) for i in key}
    tD = {i: torch.from_numpy(depth[i]).to(dev).float() for i in key}
    M3_t, mt_t = torch.tensor(M3, dtype=torch.float32, device=dev), torch.tensor(mt, dtype=torch.float32, device=dev)

    os.makedirs(a.out_dir, exist_ok=True)
    obj_path = os.path.join(a.out_dir, "relief.obj")
    mtl_lines, total_tris = [], 0
    with open(obj_path, "w") as fo:
        fo.write("mtllib relief.mtl\n")
        off = 0
        for i in key:
            z = tD[i][ys][:, xs].reshape(-1)
            Xc = torch.stack([(uf - cx) / fx * z, (vf - cy) / fy * z, z], 1)
            Xw = (Xc - tT[i]) @ tR[i]                                  # DA3 world
            Xf = Xw @ M3_t.T + mt_t                                    # final frame
            zt = z[tri]
            valid = (zt > 0).all(1)
            valid &= (zt.max(1).values / zt.min(1).values.clamp(min=1e-6) - 1) < a.edge_tol
            pf = Xf[tri]
            nf = torch.cross(pf[:, 1] - pf[:, 0], pf[:, 2] - pf[:, 0], dim=1)
            nf = nf / nf.norm(dim=1, keepdim=True).clamp(min=1e-12)
            cf, cw = pf.mean(1), Xw[tri].mean(1)
            to_cam = tC[i] - cf
            dist = to_cam.norm(dim=1)
            cosv = (nf * to_cam).sum(1).abs() / dist.clamp(min=1e-6)
            valid &= cosv > a.min_cos
            s_own = cosv ** 2 / (0.2 + dist)
            best_other = torch.zeros_like(s_own)
            for j in key:
                if j == i:
                    continue
                Xj = cw @ tR[j].T + tT[j]
                zj = Xj[:, 2]
                zs = zj.clamp(min=1e-6)
                u, v = fx * Xj[:, 0] / zs + cx, fy * Xj[:, 1] / zs + cy
                ok = (zj > 1e-3) & (u > 0.02 * W) & (u < 0.98 * W) & (v > 0.02 * H) & (v < 0.98 * H)
                dj = tD[j][v.long().clamp(0, H - 1), u.long().clamp(0, W - 1)]
                ok &= (dj > 0) & ((dj - zj).abs() < a.depth_tol * zj)
                tc = tC[j] - cf
                dj_m = tc.norm(dim=1)
                cj = (nf * tc).sum(1).abs() / dj_m.clamp(min=1e-6)
                ok &= cj > a.min_cos
                best_other = torch.maximum(best_other, torch.where(ok, cj ** 2 / (0.2 + dj_m), torch.zeros_like(cj)))
            keep = valid & (s_own >= a.slack * best_other)
            tk = tri[keep]
            if tk.numel() == 0:
                continue
            used, inv = torch.unique(tk, return_inverse=True)
            V = Xf[used].cpu().numpy()
            UV = torch.stack([uf[used] / W, 1 - vf[used] / H], 1).cpu().numpy()
            F = inv.cpu().numpy() + off + 1
            tag = f"{i:04d}"
            fo.write(f"o kf_{tag}\nusemtl m_{tag}\n")
            np.savetxt(fo, V, fmt="v %.4f %.4f %.4f")
            np.savetxt(fo, UV, fmt="vt %.5f %.5f")
            np.savetxt(fo, np.stack([F[:, 0], F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2]], 1), fmt="f %d/%d %d/%d %d/%d")
            off += len(V)
            total_tris += len(tk)
            ext = os.path.splitext(names[i])[1]
            shutil.copy(os.path.join(fdir, names[i]), os.path.join(a.out_dir, f"kf_{tag}{ext}"))
            mtl_lines += [f"newmtl m_{tag}", "Ka 1 1 1", "Kd 1 1 1", "illum 1", f"map_Kd kf_{tag}{ext}", ""]
    with open(os.path.join(a.out_dir, "relief.mtl"), "w") as fm:
        fm.write("\n".join(mtl_lines))
    print(f"[{time.time() - t0:.0f}s] wrote {obj_path}: {len(key)} pieces, {total_tris} triangles", flush=True)

    # held-out camera poses (Blender convention) for objective evaluation
    def make_views(ids, piece=False):
        out = []
        for k in ids:
            Rb = (Rot @ R[k].T) @ np.diag([1.0, -1.0, -1.0])
            im = cv2.imread(os.path.join(fdir, names[k]))
            v = dict(name=names[k], frame=os.path.abspath(os.path.join(fdir, names[k])), location=C_f[k].tolist(),
                     rotation=Rb.tolist(), angle=2 * math.atan(max(W, H) / (2 * fx)), W=int(im.shape[1]), H=int(im.shape[0]))
            if piece:  # render only this keyframe's own piece: isolates the camera mapping from neighbouring pieces
                v["piece"] = f"kf_{k:04d}"
            out.append(v)
        return out
    views = make_views(ho)
    with open(os.path.join(a.out_dir, "views.json"), "w") as f:
        json.dump(views, f, indent=1)
    # sanity check set: from a keyframe's OWN pose the render must match its frame closely, else the camera mapping is wrong
    with open(os.path.join(a.out_dir, "views_keyframes.json"), "w") as f:
        json.dump(make_views([key[len(key) // 4], key[len(key) // 2], key[3 * len(key) // 4]], piece=True), f, indent=1)
    print(f"[{time.time() - t0:.0f}s] wrote {len(views)} held-out views (views.json) + 3 keyframe sanity views", flush=True)


if __name__ == "__main__":
    main()
