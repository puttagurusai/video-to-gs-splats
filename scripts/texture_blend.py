"""Blended multi-view texturing (spec section 7.1): top-k views per texel, quality weights, exposure compensation.

Single best-view texturing (the removed texture_mesh.py) gave blocky patches where neighbouring frames differ in exposure or are a
sub-pixel off. Here every texel is a weighted average of its best K visible frames:
  * visibility per texel by a depth test against the frame's depth map (occluded views do not contribute);
  * weight = (score / best score)^power, score = cos^2 / (0.2 + distance) x sharpness, views below 30% of the best are dropped;
  * per-frame, per-channel exposure gains are solved from the frames' own overlaps (robust least squares in log space).

  python -I scripts/texture_blend.py --mesh export/mine_c_uv.obj --cams export/mine_c_cams.npz --out export/mine_c_blend.obj

Inputs: a UV-unwrapped OBJ (uv_unwrap.py) and the <name>_cams.npz bundle written by fuse_aligned.py / da3_reconstruct.py. Averaging misaligned views softens fine detail slightly; that is the price of removing
patches. Mirrors/glass and badly wrong poses are not handled.
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, vstack, identity
from scipy.sparse.linalg import lsqr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--atlas", type=int, default=4096)
    ap.add_argument("--blend", type=int, default=3, help="views blended per texel")
    ap.add_argument("--power", type=float, default=4.0, help="sharpness of the weight falloff (higher = closer to best-view)")
    ap.add_argument("--min-cos", type=float, default=0.25)
    ap.add_argument("--depth-tol", type=float, default=0.08)
    ap.add_argument("--no-gain", action="store_true", help="skip exposure compensation")
    ap.add_argument("--gain-samples", type=int, default=300000)
    a = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = np.load(a.cams)
    names = [str(x) for x in B["names"]]
    R, T = B["R"].astype(np.float32), B["t"].astype(np.float32)
    K = B["K"].astype(np.float32)
    H, W = [int(x) for x in B["hw"]]
    depth, M = B["depth"], B["world_to_mesh"].astype(np.float64)
    fdir = a.frames or str(B["frames_dir"])
    nC, k = len(names), a.blend
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    imgs, sharp = [], []
    for n in names:
        bgr = cv2.imread(os.path.join(fdir, n))
        if bgr is None:
            raise SystemExit(f"frame not found: {os.path.join(fdir, n)}")
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        sharp.append(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    sharp = np.asarray(sharp)
    sharp_w = torch.tensor(np.sqrt(sharp / np.median(sharp)).clip(0.3, 1.5), dtype=torch.float32, device=dev)

    mesh = trimesh.load(a.mesh, process=False)   # no maintain_order: vertices must split along UV seams
    if isinstance(mesh, trimesh.Scene):
        raise SystemExit("OBJ loaded as a scene; expected a single mesh with UVs.")
    V, F, UV = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), np.asarray(mesh.visual.uv, dtype=np.float64)
    nF = len(F)
    if len(UV) != len(V):
        raise SystemExit("UV count differs from vertex count; OBJ not loaded with split seams.")
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1) / 2
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    print(f"[{time.time() - t0:.0f}s] {nC} frames, mesh {len(V)} vertices / {nF} faces; blending {k} views per texel; device {dev}", flush=True)

    # ---- rasterise UV triangles, barycentric 3D position per texel ----
    A = a.atlas
    pts = np.round(np.stack([UV[F][..., 0] * A, (1 - UV[F][..., 1]) * A], -1)).astype(np.int32)
    idmap = np.full((A, A), -1, dtype=np.float32)
    for fi in range(nF):
        cv2.fillConvexPoly(idmap, pts[fi], float(fi))
    ys, xs = np.nonzero(idmap >= 0)
    fid = idmap[ys, xs].astype(np.int64)
    nT = len(fid)
    print(f"[{time.time() - t0:.0f}s] atlas {A}x{A}: {nT} texels covered ({100 * nT / (A * A):.0f}%), "
          f"~{1000 * float(np.sqrt(area.sum() / max(nT, 1))):.1f} mm per texel", flush=True)

    Vt = torch.tensor(V, dtype=torch.float32, device=dev)
    Ft = torch.tensor(F, dtype=torch.long, device=dev)
    UVt = torch.tensor(UV, dtype=torch.float32, device=dev)
    nrm_t = torch.tensor(nrm, dtype=torch.float32, device=dev)
    fid_t = torch.tensor(fid, device=dev)
    P_all = torch.empty(nT, 3, device=dev)
    CH = 2_000_000
    for s in range(0, nT, CH):
        f = fid_t[s:s + CH]
        xt = torch.tensor(xs[s:s + CH], device=dev).float()
        yt = torch.tensor(ys[s:s + CH], device=dev).float()
        uvt = torch.stack([(xt + 0.5) / A, 1 - (yt + 0.5) / A], 1)
        tu = UVt[Ft[f]]
        e0, e1, e2 = tu[:, 1] - tu[:, 0], tu[:, 2] - tu[:, 0], uvt - tu[:, 0]
        den = e0[:, 0] * e1[:, 1] - e1[:, 0] * e0[:, 1]
        den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
        l1 = (e2[:, 0] * e1[:, 1] - e1[:, 0] * e2[:, 1]) / den
        l2 = (e0[:, 0] * e2[:, 1] - e2[:, 0] * e0[:, 1]) / den
        lam = torch.stack([1 - l1 - l2, l1, l2], 1).clamp(0, 1)
        lam = lam / lam.sum(1, keepdim=True).clamp(min=1e-9)
        P_all[s:s + CH] = (lam[:, :, None] * Vt[Ft[f]]).sum(1)
    n_tex = nrm_t[fid_t]
    Minv = np.linalg.inv(M)
    P_da3 = P_all @ torch.tensor(Minv[:3, :3].astype(np.float32).T, device=dev) + torch.tensor(Minv[:3, 3].astype(np.float32), device=dev)
    Mrot, mtr = torch.tensor(M[:3, :3], dtype=torch.float32, device=dev), torch.tensor(M[:3, 3], dtype=torch.float32, device=dev)

    # ---- per texel: keep the best k visible views ----
    top_s = torch.zeros(nT, k, device=dev)
    top_i = torch.full((nT, k), -1, dtype=torch.long, device=dev)
    margin = 0.02
    for i in range(nC):
        Ri, ti = torch.tensor(R[i], device=dev), torch.tensor(T[i], device=dev)
        Xc = P_da3 @ Ri.T + ti
        z = Xc[:, 2]
        zs = z.clamp(min=1e-6)
        u, v = fx * Xc[:, 0] / zs + cx, fy * Xc[:, 1] / zs + cy
        ok = (z > 1e-3) & (u > margin * W) & (u < (1 - margin) * W) & (v > margin * H) & (v < (1 - margin) * H)
        d = torch.from_numpy(depth[i]).to(dev).float()[v.long().clamp(0, H - 1), u.long().clamp(0, W - 1)]
        ok &= (d > 0) & ((d - z).abs() < a.depth_tol * z)
        C_f = Mrot @ (-(Ri.T @ ti)) + mtr
        to_cam = C_f - P_all
        dist = to_cam.norm(dim=1)
        cosv = (n_tex * to_cam).sum(1).abs() / dist.clamp(min=1e-6)
        ok &= cosv > a.min_cos
        s = torch.where(ok, cosv ** 2 / (0.2 + dist) * sharp_w[i], torch.zeros_like(cosv))
        mn, mi = top_s.min(1)
        rows = (s > mn).nonzero().squeeze(1)
        if rows.numel():
            top_s[rows, mi[rows]] = s[rows]
            top_i[rows, mi[rows]] = i
        del Xc, z, zs, u, v, ok, d, to_cam, dist, cosv, s
    smax = top_s.max(1, keepdim=True).values.clamp(min=1e-12)
    rel = top_s / smax
    w = torch.where((rel > 0.3) & (top_i >= 0), rel ** a.power, torch.zeros_like(rel))
    nviews = (w > 0).sum(1)
    print(f"[{time.time() - t0:.0f}s] views chosen. texels with 0 / 1 / >=2 usable views: "
          f"{int((nviews == 0).sum())} / {int((nviews == 1).sum())} / {int((nviews >= 2).sum())}", flush=True)

    def sample(i, rows_idx):
        """Colours (n,3), 0..255, of frame i at the texels `rows_idx`."""
        Ri, ti = torch.tensor(R[i], device=dev), torch.tensor(T[i], device=dev)
        Xc = P_da3[rows_idx] @ Ri.T + ti
        zs = Xc[:, 2].clamp(min=1e-6)
        gx = 2 * (fx * Xc[:, 0] / zs + cx) / W - 1
        gy = 2 * (fy * Xc[:, 1] / zs + cy) / H - 1
        img = torch.from_numpy(imgs[i]).to(dev).permute(2, 0, 1)[None].float()
        return torch.nn.functional.grid_sample(img, torch.stack([gx, gy], 1)[None, None], mode="bilinear",
                                               padding_mode="border", align_corners=False)[0, :, 0, :].T

    # ---- exposure compensation: solve log-gains so frames agree on shared texels ----
    gain = torch.ones(nC, 3, device=dev)
    if not a.no_gain:
        cand = (nviews >= 2).nonzero().squeeze(1)
        if cand.numel() > 1000:
            g = torch.Generator(device="cpu").manual_seed(0)
            sub = cand[torch.randperm(cand.numel(), generator=g)[:a.gain_samples].to(dev)]
            colS = torch.zeros(len(sub), k, 3, device=dev)
            for i in range(nC):
                r, sl = ((top_i[sub] == i) & (w[sub] > 0)).nonzero(as_tuple=True)
                if r.numel():
                    colS[r, sl] = sample(i, sub[r])
            ref = w[sub].argmax(1)
            ia_list, ib_list, la_list, lb_list = [], [], [], []
            ar = torch.arange(len(sub), device=dev)
            for sl in range(k):
                m = (w[sub, sl] > 0) & (ref != sl)
                ca, cb = colS[ar, ref], colS[ar, sl]
                m &= (ca.min(1).values > 12) & (ca.max(1).values < 243) & (cb.min(1).values > 12) & (cb.max(1).values < 243)
                ia_list.append(top_i[sub, ref][m]); ib_list.append(top_i[sub][m, sl])
                la_list.append(torch.log(ca[m])); lb_list.append(torch.log(cb[m]))
            ia, ib = torch.cat(ia_list).cpu().numpy(), torch.cat(ib_list).cpu().numpy()
            la, lb = torch.cat(la_list).cpu().numpy(), torch.cat(lb_list).cpu().numpy()
            neq = len(ia)
            if neq > 500:
                rows = np.concatenate([np.arange(neq), np.arange(neq)])
                Aeq = coo_matrix((np.concatenate([np.ones(neq), -np.ones(neq)]), (rows, np.concatenate([ia, ib]))), shape=(neq, nC)).tocsr()
                lam_reg = 0.05 * np.sqrt(neq / nC)
                Areg = identity(nC, format="csr") * lam_reg
                x = np.zeros((nC, 3))
                for ch in range(3):
                    y = lb[:, ch] - la[:, ch]
                    wr = np.ones(neq)
                    xc = np.zeros(nC)
                    for _ in range(4):  # IRLS: down-weight texels where frames disagree for other reasons (occlusion, highlights)
                        Aw = vstack([Aeq.multiply(wr[:, None]).tocsr(), Areg])
                        yw = np.concatenate([y * wr, np.zeros(nC)])
                        xc = lsqr(Aw, yw, atol=1e-8, btol=1e-8, iter_lim=300)[0]
                        res = np.abs(Aeq @ xc - y)
                        wr = 1.0 / np.maximum(1.0, res / 0.15)
                    x[:, ch] = xc
                gain = torch.tensor(np.exp(x).clip(0.7, 1.4), dtype=torch.float32, device=dev)
                gmin, gmax = gain.min().item(), gain.max().item()
                print(f"[{time.time() - t0:.0f}s] exposure gains from {neq} texel pairs: range {gmin:.2f} .. {gmax:.2f}", flush=True)
            else:
                print("too few overlapping texels for exposure compensation; gains = 1", flush=True)
        del cand

    # ---- blend ----
    acc = torch.zeros(nT, 3, device=dev)
    wsum = torch.zeros(nT, device=dev)
    for i in range(nC):
        r, sl = ((top_i == i) & (w > 0)).nonzero(as_tuple=True)
        if r.numel() == 0:
            continue
        col = sample(i, r) * gain[i]
        ww = w[r, sl]
        acc.index_add_(0, r, col * ww[:, None])
        wsum.index_add_(0, r, ww)
    valid_t = wsum > 0
    col_t = torch.where(valid_t[:, None], acc / wsum.clamp(min=1e-9)[:, None], torch.zeros_like(acc)).clamp(0, 255)
    atlas = np.zeros((A, A, 3), dtype=np.uint8)
    valid = np.zeros((A, A), dtype=bool)
    vt = valid_t.cpu().numpy()
    atlas[ys[vt], xs[vt]] = col_t[valid_t].byte().cpu().numpy()
    valid[ys[vt], xs[vt]] = True
    print(f"[{time.time() - t0:.0f}s] blended; {100 * vt.mean():.1f}% of covered texels have colour; padding the rest", flush=True)

    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    atlas = atlas[iy, ix]
    img_out = Image.fromarray(atlas)
    stem = os.path.splitext(os.path.basename(a.out))[0]
    mat = trimesh.visual.material.SimpleMaterial(image=img_out)
    mat.name = stem
    out = trimesh.Trimesh(vertices=V, faces=F, process=False, visual=trimesh.visual.TextureVisuals(uv=UV, material=mat))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.export(a.out, mtl_name=stem + ".mtl")
    img_out.save(os.path.splitext(a.out)[0] + "_atlas.png")
    print(f"[{time.time() - t0:.0f}s] done. wrote {a.out} (+ .mtl/.png)", flush=True)


if __name__ == "__main__":
    main()
