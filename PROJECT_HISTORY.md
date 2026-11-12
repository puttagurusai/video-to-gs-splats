# Project history: how we got here, what we tried, and why

`README.md` explains **what the project is and how to set it up and run it**. This document explains **how it was
built**: the steps, the decisions, the dead ends and the lessons. Read the README first, then this file if you need to
know why something is the way it is, or before repeating an experiment.

Numbers come from `RUNLOG.md` (measured on this laptop: RTX 500 Ada 4 GB, 22 logical CPUs, 63 GB RAM, one drive).
"Held-out SSIM" is the quality score defined in the README (section 6). Only two clips were ever tested: the **mouse**
clip (10 s close-up, 125 frames) and the **hotel** clip (14 s walkthrough, 142 frames).

---

## 1. Goal and how it changed

| Phase | Goal | What changed it |
|---|---|---|
| Start | Follow the written spec (`Video_to_3D_Complete_Pipeline_and_Depth_Alignment.docx`): video to a playable 3D room using COLMAP, OpenMVS, Blender and Godot. | The full tool chain needed far more disk and a bigger GPU than this laptop has (a list of required tools and sizes was requested and given). |
| Pivot 1 | Fast, low-resource, general (nothing hard-coded to one clip) reconstruction. | Runs took over 30 minutes for a 30 s video; extrapolated, a 3-minute video would take hours. |
| Pivot 2 | Output must look like the video ("at least 80 % realistic"). | The depth-based mesh did not look like the video. Gaussian splats did. |
| Pivot 3 | Godot dropped; output is viewed in Brush (splat) or Blender (mesh). | Not needed for the goal. |
| Now | A one-command, one-double-click pipeline with a mesh mode kept as an option, and faster poses. | See sections 3 and 5. |

Standing preferences from the owner of the project: minutes, not hours; fixes must be general; ask before downloads; no
long speculative test runs.

---

## 2. Chronology of the work

1. **Spec and environment check.** Listed which tools were already present and which were missing, and why the full
   chain needs about 50 GB. Found the limits: 4 GB VRAM, little free disk.
2. **COLMAP on this laptop.** The official CUDA build aborts inside CUDA initialisation (exit code 0xC0000409) for both
   SIFT extraction and matching; root cause never found. The CPU incremental mapper was slow and fragmented on the first
   clip. COLMAP's GLOMAP-style `global_mapper` is faster, and works on the mouse clip.
3. **Depth odometry route** (`depth_odometry.py`): Depth-Anything-V2 metric depth, GPU matching, PnP, pose graph, TSDF.
   About 75 s for 142 frames. Limits: absolute scale needs a measured reference; self-calibrated focal length was
   unreliable.
4. **Depth Anything 3 (DA3) backend** (`da3_reconstruct.py`): poses, intrinsics and depth from one model, run in chunks
   (2.5 GB peak) and stitched with Sim3. About 103 s for the hotel clip. DA3's separate fx/fy proved unreliable, so a
   photometric square-pixel focal calibration was added. Metric scale from the helper depth model was not credible
   (it implied a 5.9 m ceiling), so scale is left as measured-by-the-user.
5. **First look at quality.** The mesh from the mouse clip "did not look realistic". The quality bar was set at 80 %.
6. **Spec path with COLMAP/GLOMAP poses.** Because COLMAP gives a correct, calibrated focal length, the spec pipeline
   was built on it: joint multi-view depth alignment (`depth_align.py`, `align_real.py`, with a synthetic test),
   plane-sweep depth refinement (`refine_depth.py`), GPU TSDF fusion (`fuse_aligned.py`), UV unwrap in Blender
   (`uv_unwrap.py`), blended multi-view texturing with exposure compensation (`texture_blend.py`).
7. **How quality was measured, and a correction.** Held-out views were rendered from the mesh and compared with the real
   frames (`relief_scene.py`, `render_views.py`, `eval_views.py`). Two evaluation bugs were found and fixed (an
   anisotropic DA3 intrinsic matrix, and neighbouring mesh pieces occluding an isolated-keyframe test). The mesh
   "held-out" frames were later disclosed as **not truly held out** (they were used for fusion and texturing), so the
   mesh scores are optimistic.
   Mesh progress (raw SSIM / coarse SSIM): single view 0.414 / 0.604, blended 0.427 / 0.622, refined and blended 0.458 / 0.649.
   Conclusion: far below the 80 % target; the limit is pixel-level misregistration and flat monocular geometry.
8. **Gaussian splatting.** A pure-PyTorch splat code from a sibling project was rejected (it did not produce correct
   output). `gsplat` was judged not feasible (no `nvcc` or MSVC, about 3 GB of disk). **Brush v0.3.0** (WebGPU, no CUDA
   toolchain) trained from the undistorted COLMAP folder in 124 s with about 1.1 GB VRAM: true held-out SSIM **0.857**
   vs **0.46** for the mesh. This became the default output.
9. **One-command pipeline** (`run_splat.py`): frames, COLMAP/global mapper, gates, DA3 fallback, Brush, held-out scoring.
   Mouse clip 280 s, SSIM 0.845, GOOD. Hotel clip: COLMAP ran 254 s and was then **rejected by the flip gate** (depth-flipped
   model), DA3 poses were used, total 454 s, SSIM 0.726, FAIR.
10. **DA3 pose refinement** (`refine_poses.py`): COLMAP triangulation plus bundle adjustment on existing matches. Round 1
    gave a small real gain (hotel SSIM 0.726 to 0.738); round 2 diverged and a guard now discards it. Not part of the default pipeline.
11. **Launcher and housekeeping.** `run_video_to_splat.bat` (all settings at the top, estimated progress bar), cleanup of
    unused files (2.5 GB), a CPU-only COLMAP copy placed inside the project, `MODE` switch (splat / mesh / both),
    frame budget (`FPS`, `MAX_FRAMES`, `MIN_FPS`), optional `TIME_BUDGET_MIN`, `RESUME`.
12. **Speed investigation** (section 5).
13. **Final cleanup (2026-11-08):** removed legacy scripts and temporary files and wrote the README.

---

## 3. Key decisions and the reason for each

| Decision | Reason |
|---|---|
| Splat is the default, mesh is optional | Held-out SSIM 0.85 vs 0.46 on the same clip; splats look like the video. A mesh is only needed for Blender. |
| Brush instead of gsplat or other CUDA trainers | No compiler on this machine; Brush runs on WebGPU, Apache-2.0, 1 GB VRAM. |
| COLMAP poses first, DA3 as fallback | Controlled test on the mouse clip: COLMAP poses 0.86 SSIM vs DA3 poses 0.39. DA3 is faster but much less accurate for splats. |
| Global mapper (GLOMAP-style) instead of the incremental mapper | Much faster; the incremental mapper was slow and fragmented on the first clip. |
| Two gates on COLMAP output: registration fraction and depth-flip check | A model can reproject well and still be depth-flipped (hotel clip); the flip check compares sparse depth with Depth-Anything-V2 relative depth. |
| CPU-only COLMAP in `tools\colmap-cpu` | The CUDA build crashes; a local copy keeps the project self-contained. `colmap-cuda` is kept on request. |
| Square-pixel photometric focal calibration for DA3 | DA3's fx and fy differ and were unreliable; the photometric estimate matched COLMAP on the mouse clip (1.00 vs 0.99 x long side). |
| Depth-based start point cloud for DA3 datasets | Empty start cloud gave SSIM 0.633 vs 0.726 on the hotel clip. |
| Held-out frames excluded from training | The only honest quality number available without ground truth. |
| Heuristic verdict GOOD / FAIR / POOR at SSIM 0.80 / 0.60 | Derived from our two clips; not a standard. |
| DA3-BASE only | Only SMALL and BASE are Apache-2.0; the larger checkpoints are non-commercial. Legal should confirm licences. |

---

## 4. Problems met and how they were fixed

| Problem | Cause | Fix |
|---|---|---|
| COLMAP CUDA build crashes | Unknown (aborts in CUDA init) | CPU build; `run_sfm.py` probes GPU and falls back |
| Hotel model depth-flipped | Global mapper local optimum | `sfm_check.py` gate, DA3 fallback |
| Model download failed with `WinError 1314` and the wrapper still returned success | No symlink privilege for the Hugging Face cache; wrapper exit code hid it | Download with `local_dir`; always read real exit codes |
| UVs scrambled | `trimesh` `maintain_order=True` | Load OBJ without it; check UV and vertex counts |
| Blender could not find textures | Relative paths | Absolute paths in helper scripts |
| `mesh.split()` very slow on noisy meshes | Many tiny components | scipy connected-components filter |
| `point_triangulator` aborted | COLMAP 4.x rigs/frames: database camera and image IDs must match the model | `match_database_camera`, `build_input_model` in `refine_poses.py` |
| Bundle-adjustment round 2 diverged | Unknown | Guard keeps the last valid round |
| Evaluation looked wrong | Anisotropic intrinsics, occlusion by neighbouring pieces | Square-pixel focal; isolated-piece render |
| Bat file test failed | The video had been moved into `video\` | Defaults repointed |
| COLMAP 4.2 reads old-format models badly | Missing rigs/frames files | Convert via text and `model_converter` |

Windows notes: PowerShell 5.1 mangles stderr from native programs and file encodings; Blender needs absolute paths;
the batch pattern `set "X=v" & REM comment` is verified; `python -I` avoids importing stray local files.

---

## 5. Speed investigation

Profile (mouse, 125 frames): COLMAP 171 s (extract 20, match 92, mapper about 50), flip check 26, Brush 74, other 8.
COLMAP is 55-65 % of a run; bundle adjustment is CPU in every COLMAP build.

* **GPU question.** SIFT extraction and matching could use the GPU, but the CUDA build crashes here. Brush already uses
  the GPU (about 1 GB of 4 GB); its utilisation was not measured.
* **Matching bottleneck (measured).** During CPU matching the machine sits at about 10 % CPU (roughly 2 of 22 cores);
  `--FeatureMatching.num_threads` has no effect. `scripts\parallel_match.py` splits the pairs over several COLMAP
  processes and merges the databases. **It was written but never successfully tested** (the test run was interrupted).
* **FastMap** (GPU global SfM replacing only the mapper): cloned and run on Windows through its PyTorch path after a
  one-line patch for COLMAP 4.2 databases. 209 s vs about 50 s for COLMAP's mapper, poses about 13 degrees different
  (not judged), no Windows port or CUDA kernels. **Dropped** and deleted.
* **Not tried:** COLMAP's built-in ALIKED and LightGlue (needs a model download), InstantSfM, VGGT-X / InstantSplat
  (memory needs unknown), running DA3 in parallel with COLMAP, starting the flip check during Brush training.

---

## 6. Lessons

1. Measure before claiming: an early claim that GPU matching would save "very little" was wrong and was corrected with a profile.
2. A low reprojection error does not prove a good model; always add an independent check (the flip gate).
3. Hold out views for real, and say so when an evaluation is leaky.
4. Pose quality dominates splat quality: poses from a faster method cost far more quality than they save in time.
5. Check real exit codes; wrappers can hide failures.
6. Keep fixes general (settings and gates), not tuned to one clip.

---

## 7. Open items

See README section 11 for the full list. In short: the bat file and mesh mode have not been run end to end through
the orchestrator; 2-minute-video timings are extrapolated; `parallel_match.py` and the other speed-ups are untested;
the hotel clip only reaches FAIR because COLMAP failed on it; the cause of the CUDA crash is unknown.

Reproducibility (versions, hashes, commands, all numbers): `RUNLOG.md`.
