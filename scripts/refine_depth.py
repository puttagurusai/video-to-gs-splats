"""Multi-view depth refinement: make per-frame depth agree with the neighbouring frames (small plane sweep on the GPU).

DA3 depth is smooth and plausible but each frame's scale/shape is slightly off, so surfaces from different frames do
not coincide. For every pixel we try a range of depth hypotheses around the DA3 value (default 0.8x-1.25x), warp
nearby frames into this one using the camera poses, and keep the hypothesis whose image patch matches best (ZNCC).
Low-confidence pixels (textureless, occluded, no baseline) stay at the DA3 value; the correction map is median-filtered.

Run with the DA3 venv python (needs torch):
  tools\\venv_da3\\Scripts\\python.exe -I scripts\\refine_depth.py --cams export\\room_t_cams.npz --out export\\room_t_ref_cams.npz

Writes a copy of the bundle with refined depth (same layout; feed it to relief_scene.py or texture_blend.py).
Limits: needs real parallax between neighbours; repetitive or blank texture gives no information (those pixels keep
the DA3 depth); poses that are wrong cannot be fixed here.
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--hyp", type=int, default=21, help="number of depth hypotheses")
    ap.add_argument("--range", type=float, nargs=2, default=[0.8, 1.25], help="hypothesis range, relative to DA3 depth")
    ap.add_argument("--neighbors", type=int, default=4)
    ap.add_argument("--window", type=int, default=7, help="matching patch size (px at DA3 resolution)")
    ap.add_argument("--max-gap", type=int, default=10, help="neighbour frames searched within +-this many frames")
    ap.add_argument("--min-base", type=float, default=0.02, help="min baseline / depth for a useful neighbour")
    ap.add_argument("--max-base", type=float, default=0.35)
    ap.add_argument("--conf-lo", type=float, default=0.35, help="ZNCC at which the correction starts to be trusted")
    ap.add_argument("--conf-hi", type=float, default=0.75, help="ZNCC at which the correction is fully trusted")
    a = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = np.load(a.cams, allow_pickle=False)
    data = {k: B[k] for k in B.files}
    names = [str(x) for x in data["names"]]
    R, T = data["R"].astype(np.float32), data["t"].astype(np.float32)
    K = data["K"].astype(np.float32)
    H, W = [int(x) for x in data["hw"]]
    depth = data["depth"].astype(np.float32)
    fdir = a.frames or str(data["frames_dir"])
    n = len(names)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    gray = []
    for nm in names:
        im = cv2.imread(os.path.join(fdir, nm), cv2.IMREAD_GRAYSCALE)
        gray.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)  # DA3 does a pure resize
    G = torch.tensor(np.stack(gray), device=dev)[:, None]                       # (n,1,H,W)
    Rt, Tt = torch.tensor(R, device=dev), torch.tensor(T, device=dev)
    C = -torch.einsum("nji,nj->ni", Rt, Tt)                                      # camera centres, DA3 world
    scales = torch.exp(torch.linspace(np.log(a.range[0]), np.log(a.range[1]), a.hyp, device=dev))
    vv, uu = torch.meshgrid(torch.arange(H, device=dev).float() + 0.5, torch.arange(W, device=dev).float() + 0.5, indexing="ij")
    rays = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], -1)  # (H,W,3)
    pad = a.window // 2

    def zncc(x, y):  # x,y: (...,1,H,W); local zero-normalised cross-correlation
        mx, my = Fn.avg_pool2d(x, a.window, 1, pad), Fn.avg_pool2d(y, a.window, 1, pad)
        vx = Fn.avg_pool2d(x * x, a.window, 1, pad) - mx * mx
        vy = Fn.avg_pool2d(y * y, a.window, 1, pad) - my * my
        cov = Fn.avg_pool2d(x * y, a.window, 1, pad) - mx * my
        return cov / torch.sqrt(vx.clamp(min=1e-5) * vy.clamp(min=1e-5))

    out_depth = depth.copy()
    stats = []
    for i in range(n):
        D = torch.tensor(depth[i], device=dev)
        if (D > 0).float().mean() < 0.05:
            continue
        zmed = float(D[D > 0].median())
        cand = []
        for j in range(max(0, i - a.max_gap), min(n, i + a.max_gap + 1)):
            if j == i:
                continue
            base = float((C[j] - C[i]).norm()) / zmed
            if a.min_base <= base <= a.max_base:
                cand.append((abs(base - 0.1), j))   # prefer a baseline near 10% of the depth
        nbrs = [j for _, j in sorted(cand)[:a.neighbors]]
        if not nbrs:
            continue
        # hypotheses: (M,H,W) depth; world points; project into each neighbour
        Zh = D[None] * scales[:, None, None]
        Xc = rays[None] * Zh[..., None]                                          # (M,H,W,3) camera i
        Xw = (Xc - Tt[i]) @ Rt[i]                                                # world
        gi = G[i][None].expand(len(scales), -1, -1, -1)
        score_sum = torch.zeros(len(scales), 1, H, W, device=dev)
        count = torch.zeros(1, 1, H, W, device=dev)
        for j in nbrs:
            Xj = Xw @ Rt[j].T + Tt[j]
            zj = Xj[..., 2]
            ok = zj > 1e-3
            u = fx * Xj[..., 0] / zj.clamp(min=1e-6) + cx
            v = fy * Xj[..., 1] / zj.clamp(min=1e-6) + cy
            ok &= (u > 0) & (u < W) & (v > 0) & (v < H)
            grid = torch.stack([2 * u / W - 1, 2 * v / H - 1], -1)               # (M,H,W,2)
            warped = Fn.grid_sample(G[j][None].expand(len(scales), -1, -1, -1), grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            sc = zncc(gi, warped)
            okf = ok[:, None].float()
            score_sum += sc * okf
            count += okf.amax(0, keepdim=True)
        score = score_sum / count.clamp(min=1)                                   # (M,1,H,W) mean ZNCC over valid neighbours
        best, m = score.max(0)                                                   # (1,H,W)
        s_best = scales[m[0]]                                                    # (H,W)
        conf = ((best[0] - a.conf_lo) / (a.conf_hi - a.conf_lo)).clamp(0, 1) * (count[0, 0] > 0).float()
        s_np = cv2.medianBlur(s_best.cpu().numpy().astype(np.float32), 5)
        c_np = cv2.medianBlur(conf.cpu().numpy().astype(np.float32), 5)
        s_final = 1.0 + c_np * (s_np - 1.0)
        out_depth[i] = np.where(depth[i] > 0, depth[i] * s_final, 0)
        valid = depth[i] > 0
        stats.append((float(np.mean(np.abs(s_final[valid] - 1))), float((c_np[valid] > 0.5).mean())))
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  refined {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    if stats:
        print(f"mean |scale correction| {100 * np.mean([s[0] for s in stats]):.1f}%; "
              f"confident pixels {100 * np.mean([s[1] for s in stats]):.0f}% (rest keep the DA3 depth); frames refined {len(stats)}/{n}")
    data["depth"] = out_depth.astype(np.float16)
    np.savez_compressed(a.out, **data)
    print(f"[{time.time() - t0:.0f}s] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
