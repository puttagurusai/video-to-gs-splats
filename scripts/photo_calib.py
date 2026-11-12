"""Photometric focal-length calibration (square pixels), shared by da3_reconstruct.py and export_colmap_dataset.py.

DA3 predicts fx and fy separately and they can differ by ~20% (a real phone camera has square pixels). Using both as
given distorts the lifted geometry. Here the focal length is chosen by multi-view agreement: lift frame i with its depth,
project into a neighbouring frame j with the pose, and measure local ZNCC. A wrong focal length lowers the agreement.
"""
import numpy as np


def calibrate_isotropic_focal(depth, R, T, cx, cy, gray, device, n_pairs=20, window=7):
    """depth (n,H,W) float32 (0 = invalid, DA3 units); R (n,3,3), T (n,3) world->camera; gray (n,H,W) float32 in 0..1.
    Returns (f_best_px, [(f_px, agreement), ...])."""
    import torch
    import torch.nn.functional as Fn
    n, H, W = depth.shape
    d = torch.tensor(depth, dtype=torch.float32, device=device)
    Rt, Tt = torch.tensor(R, dtype=torch.float32, device=device), torch.tensor(T, dtype=torch.float32, device=device)
    G = torch.tensor(gray, dtype=torch.float32, device=device)[:, None]
    C = -torch.einsum("nji,nj->ni", Rt, Tt)
    vv, uu = torch.meshgrid(torch.arange(H, device=device).float() + 0.5, torch.arange(W, device=device).float() + 0.5, indexing="ij")
    pad = window // 2

    def zncc(x, y):
        mx, my = Fn.avg_pool2d(x, window, 1, pad), Fn.avg_pool2d(y, window, 1, pad)
        vx = Fn.avg_pool2d(x * x, window, 1, pad) - mx * mx
        vy = Fn.avg_pool2d(y * y, window, 1, pad) - my * my
        return (Fn.avg_pool2d(x * y, window, 1, pad) - mx * my) / torch.sqrt(vx.clamp(min=1e-5) * vy.clamp(min=1e-5))

    pairs = []
    for i in np.unique(np.linspace(0, n - 1, n_pairs).round().astype(int)):
        D = d[i]
        if (D > 0).float().mean() < 0.2:
            continue
        zmed = float(D[D > 0].median())
        best = None
        for j in range(max(0, i - 8), min(n, i + 9)):
            if j == i:
                continue
            base = float((C[j] - C[i]).norm()) / zmed
            if 0.03 <= base <= 0.35 and (best is None or abs(base - 0.1) < best[0]):
                best = (abs(base - 0.1), j)
        if best:
            pairs.append((int(i), best[1]))
    if len(pairs) < 3:
        return None, []

    def score(f):
        tot, cnt = 0.0, 0
        for i, j in pairs:
            D = d[i]
            Xc = torch.stack([(uu - cx) / f * D, (vv - cy) / f * D, D], -1)
            Xj = ((Xc - Tt[i]) @ Rt[i]) @ Rt[j].T + Tt[j]
            z = Xj[..., 2]
            u, v = f * Xj[..., 0] / z.clamp(min=1e-6) + cx, f * Xj[..., 1] / z.clamp(min=1e-6) + cy
            ok = (D > 0) & (z > 1e-3) & (u > 4) & (u < W - 4) & (v > 4) & (v < H - 4)
            w = Fn.grid_sample(G[j][None], torch.stack([2 * u / W - 1, 2 * v / H - 1], -1)[None], mode="bilinear",
                               padding_mode="zeros", align_corners=False)
            tot += float(zncc(G[i][None], w)[0, 0][ok].sum())
            cnt += int(ok.sum())
        return tot / max(cnt, 1)

    L = float(max(H, W))
    table = {}
    for fr in np.arange(0.35, 1.001, 0.05):
        table[round(float(fr * L), 1)] = score(float(fr * L))
    f0 = max(table, key=table.get)
    for fr in np.arange(f0 / L - 0.05, f0 / L + 0.0501, 0.0125):
        f = round(float(fr * L), 1)
        if f > 0.2 * L and f not in table:
            table[f] = score(f)
    best = max(table, key=table.get)
    return float(best), sorted(table.items())
