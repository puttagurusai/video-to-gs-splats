# RUNLOG (versions, sources, commands)

> Note (2026-11-08 cleanup): `scripts/blender_cleanup.py`, `texture_mesh.py`, `preview.py`, `render_preview.py`,
> `calib_focal_photo.py` and `TESTING.md` were removed; commands below that use them are historical. Current usage is in `README.md`.

## Depth Anything 3 backend (set up 2026-11-06)
- Source: https://github.com/ByteDance-Seed/Depth-Anything-3 at commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, cloned to `tools/da3_src` (not pip-installed; added to `sys.path` by `scripts/da3_reconstruct.py`).
- Venv: `tools/venv_da3` (python 3.13, `--system-site-packages`, so it reuses the system torch 2.7.1+cu118 / numpy 2.3.5).
  Extra packages installed with numpy/torch pinned by a constraints file: addict 2.4.0, omegaconf 2.4.0, moviepy 1.0.3, evo 1.38.1, plyfile 1.1.5 (+ their dependencies).
  Not installed (not needed for inference): xformers, open3d, e3nn, gsplat. The upstream `numpy<2` pin is ignored on purpose.
- Models (Apache-2.0 per Hugging Face card; downloaded with `local_dir` because this Windows account cannot create symlinks for the HF cache):
  - `tools/da3_models/DA3-BASE/model.safetensors` sha256 `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5` (516 MB)
  - `tools/da3_models/DA3-SMALL/model.safetensors` sha256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf` (131 MB)
  - NOT used: DA3-LARGE/GIANT/NESTED (CC BY-NC 4.0). DA3METRIC-LARGE (Apache-2.0, 1275 MB) not downloaded.
- Leftover: a failed partial download of DA3-SMALL sits in the user-level HF cache (`~/.cache/huggingface/hub/models--depth-anything--DA3-SMALL`, ~131 MB, no snapshot). Safe to delete.

## Commands
```
python -I scripts/extract_frames.py video/room.mp4 frames_10fps --fps 10 --max-side 1200
tools\venv_da3\Scripts\python.exe -I scripts\da3_reconstruct.py --frames frames_10fps --out export\room_da3.ply [--ceiling-height <real meters>]
blender --background --python scripts/blender_cleanup.py -- --input export/room_da3.ply --output export/room_da3.glb --collision-tris 0 --save-blend export/room_da3.blend
```

## Results on the sample clip (13394638_1080_1920_30fps, 14 s, 142 frames at 10 fps)
- DA3-BASE, chunks of 24 (auto-probed), overlap 5, process_res 504: peak GPU memory 2.47 GB, ~103 s in script.
- Chunk alignment residual 0.0086 DA3 units; chunk Sim3 scales 0.80-1.07; DA3 focal estimate 0.524 x longer side.
- DA-V2 metric scale 7.18 m per DA3 unit (spread 0.099) implies a 5.9 m ceiling: not credible. Absolute scale needs a real reference (`--ceiling-height`); the 2.7 m used here is an assumption.

## COLMAP + spec depth alignment path (2026-11-06, mouse clip `1000363360.mp4`, 125 frames at 10 fps)
```
python -I scripts/extract_frames.py 1000363360.mp4 frames_mine --fps 10 --max-side 1200
python -I scripts/run_sfm.py --frames frames_mine --work colmap_mine --fast --mapper global --focal-factor 0.72     # 218 s CPU
python -I scripts/align_real.py --work colmap_mine/dense --out export/mine_aligned.npz                              # ~2 min
python -I scripts/fuse_aligned.py --aligned export/mine_aligned.npz --out export/mine_c.ply
python -I scripts/relief_scene.py --cams export/mine_c_cams.npz --out-dir export/mine_c_relief
blender --background --python scripts/uv_unwrap.py -- --input export/mine_c.ply --output export/mine_c_uv.obj
python -I scripts/texture_mesh.py --mesh export/mine_c_uv.obj --cams export/mine_c_cams.npz --out export/mine_c_tex.obj
blender --background --python scripts/obj_to_blend.py -- --input export/mine_c_tex.obj --blend export/mine_c_tex.blend
blender --background export/mine_c_tex.blend --python scripts/render_views.py -- --views export/mine_c_relief/views.json --out-prefix export/mine_c_eval_fused
python -I scripts/eval_views.py --views export/mine_c_relief/views.json --prefix export/mine_c_eval_fused
```
- COLMAP global mapper: 125/125 registered, 29,566 points, 0.52 px; flip check Spearman -0.98 (correct); COLMAP focal 442 px at 252x448 (fx = fy).
- Depth alignment: anchor error 1.0% -> 1.4%; held-out cross-view |log ratio| median 0.0117 -> 0.0094, P90 0.0416 -> 0.0300.
- Scale: COLMAP units are arbitrary; the result is scaled so median depth = 1 ("scene units", NOT meters).
- Evaluation sanity (isolated keyframe piece, own pose): SSIM 0.87, PSNR 37 dB over 17-24% coverage: camera mapping verified.
- Held-out views (6), mean: fused textured mesh SSIM 0.414 / blurred-SSIM 0.604 / PSNR 15.9 dB; relief collage 0.378 / 0.548 / 15.2.
- `depth_align.py` synthetic test (`scripts/test_depth_align.py`): all 7 checks pass.
- Blended texture (`texture_blend.py`, top-3 views, exposure gains 0.87-1.24): held-out SSIM 0.427 / blurred 0.622 / PSNR 16.3 (single-view was 0.414 / 0.604 / 15.9).
- Depth refinement with COLMAP poses (`refine_depth.py --range 0.97 1.03 --hyp 25`, 16 s; mean correction 1.5%, 90% confident pixels), then `fuse_aligned.py`, `uv_unwrap.py`, `texture_blend.py`: held-out SSIM **0.458** / blurred **0.649** / PSNR **16.8** (4 of 6 views better, 2 slightly worse). Output: `export/mine_r_blend.blend`.
- **Gaussian splatting with Brush v0.3.0** (Apache-2.0, `tools/brush/brush_app.exe`, zip sha256 `b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6`, downloaded from github.com/ArthurBrussee/brush releases). No CUDA toolchain needed (WebGPU). Input: the undistorted COLMAP folder `colmap_mine/dense` as-is.
  Command: `tools\brush\brush_app.exe colmap_mine\dense --total-steps 6000 --max-resolution 800 --max-splats 600000 --sh-degree 2 --eval-split-every 10 --eval-every 2000 --eval-save-to-disk --export-path export\brush_run1 --export-every 2000`
  Cost: 124 s, peak GPU memory ~1.08 GB of 4 GB, output `export/brush_run1/export_6000.ply` (23.7 MB).
  Held-out quality (13 frames = every 10th, excluded from training; `scripts/eval_splat.py`, same metrics as the mesh path): SSIM **0.857**, PSNR **25.3 dB**, blurred SSIM **0.946** (per-frame SSIM range 0.74-0.91).
  Caveats: held-out frames are neighbours of training frames (standard 3DGS protocol, an interpolation test); mesh-path "held-out" frames were NOT excluded from fusion/texturing, so the mesh numbers above are optimistic and the true gap is at least as large; only the mouse clip was tested.
- **One-command splat pipeline** `scripts/run_splat.py` (frames -> COLMAP/GLOMAP + `sfm_check.py` flip gate -> DA3 fallback via `da3_reconstruct.py --poses-only` + `export_colmap_dataset.py` -> Brush -> `eval_splat.py`). End-to-end tests (4000 Brush steps, 800 px):
  - mouse clip (125 frames): COLMAP/GLOMAP accepted; total 280 s (frames 6, COLMAP 171, flip check 26, Brush 74, eval 2); peak GPU memory 964 MiB; held-out SSIM 0.845 / PSNR 24.9 dB / coarse 0.942 -> GOOD.
  - hotel clip (142 frames): COLMAP/GLOMAP ran 254 s, then the flip gate REJECTED it (median Spearman +0.947, i.e. depth-flipped); fallback to DA3 poses (55 s) + export (14 s) + Brush (92 s); total 454 s; peak GPU memory 1130 MiB; held-out SSIM 0.726 / PSNR 21.4 dB / coarse 0.890 -> FAIR.
  - Controlled comparison on the mouse clip (same Brush settings, 6000 steps): COLMAP poses SSIM 0.857 vs DA3 poses 0.394.
  - Heuristic verdict thresholds (held-out SSIM): >=0.80 GOOD, 0.60-0.80 FAIR, <0.60 POOR (from our clips, not a standard).
  - Photometric focal scan agreed with COLMAP on the mouse clip (1.00 vs 0.99 x longer side).
- Progress on this clip so far (raw SSIM / blurred SSIM): single-view fused 0.414/0.604 -> blended 0.427/0.622 -> refined+blended 0.458/0.649. Target of the user: "at least 80% realistic" (not reached).

## Cleanup and pose-refinement notes (2026-11-06, end of session)
- Cleanup: removed intermediate/failed folders (old COLMAP runs, frame sets, datasets, mesh-path export files, experiment runs, installer zips, DA3-SMALL weights, Godot scaffold): 2,472 MB freed (1.56 -> 3.98 GB free). Deliverables kept in `results/` (mouse_splat.ply, hotel_splat.ply, mouse/hotel mesh .blend/.glb, mouse_splat_vs_real.png). Paths in the older entries above that point at `export/`, `colmap_mine/`, `frames_*` no longer exist.
- COLMAP placement: `tools/colmap-cpu` (exact copy of the CPU-only 4.2.1 build, 292 files, 338.8 MB) is used first by `run_sfm.py`; `tools/colmap-cuda` kept (its GPU path crashes on this laptop: fast-fail 0xC0000409 in SIFT extraction and matching).
- DA3 pose refinement (`scripts/refine_poses.py`, COLMAP point_triangulator + bundle_adjuster on the existing matches; needed the database camera and image IDs to match the model: COLMAP 4.2 aborts otherwise): round 1 reprojection 1.83 -> 0.54 px; round 2 diverged (mean error 1e+150 px) and is now discarded by a guard.
  Hotel held-out SSIM at 4000 steps: plain DA3 poses 0.726 (seeds 42/1/2: 0.726, 0.727, 0.726 => Brush run-to-run noise about +-0.001); round-1 refined poses 0.738 (seed 1: 0.737) = small but real gain; the diverged round-2 model scored 0.781 (unexplained, real, not noise: the mean-error guard may be too strict, the poses may have improved while a few points exploded).
  Starting point cloud matters: empty cloud 0.633 (DA3 poses) / 0.637 (round-1 poses) vs 0.726 / 0.738 with the depth-based cloud.
- `run_video_to_splat.bat`: all settings at the top with comments; console shows an ESTIMATED progress bar (COLMAP and Brush report no percentage), stage, last log line.
