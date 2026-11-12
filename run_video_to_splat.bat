@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Video to 3D (Gaussian splat / depth-based mesh)
cd /d "%~dp0"

REM ======================================================================================================
REM   VIDEO -> 3D          Edit ONLY the variables in the box below, save, then double-click this file.
REM   A console window opens and shows live progress: an ESTIMATED percentage, the current stage and the
REM   last log line. Logs of every stage are kept in  runs\<NAME>\logs  and a summary in  runs\<NAME>\summary.json
REM ======================================================================================================

REM ------------------------------------------- WHAT TO MAKE -------------------------------------------
set "VIDEO=video\hotel_room.mp4"  & REM Input video (mp4, mov, ...): a full path, or a path relative to this folder (e.g. video\mouse.mp4). REQUIRED.
set "NAME=hotel_room"      & REM Output folder name -> runs\NAME\ . Leave empty to use the video file name. Must not exist yet (see RESUME / FORCE).
set "MODE=splat"           & REM splat = Gaussian splat (looks closest to the video, view in Brush) ; mesh = normal depth-based textured mesh (.blend/.glb) ; both = splat + mesh.

REM ------------------------------------------- SPLAT TRAINING (MODE=splat or both) --------------------
set "STEPS=4000"           & REM Training steps. More = slightly better and slower: 2000 ~1 min, 4000 ~1.5 min, 8000 ~3 min (on this laptop).
set "MAX_RES=800"          & REM Training image size (longer side, px). 800 is safe for a 4 GB GPU; 1000-1200 = sharper but more memory/time.
set "MAX_SPLATS=600000"    & REM Maximum number of Gaussian blobs. Lower (300000) if you run out of GPU memory; higher = more detail.
set "HELDOUT_EVERY=10"     & REM Every Nth frame is hidden from training and used to score quality (10 = about 1 frame in 10).

REM ------------------------------------------- FRAMES (how much of the video is used) -----------------
set "FPS=10"               & REM Target frames kept per second of video. 10 suits a slow walk or orbit; lower is faster.
set "MAX_FRAMES=400"       & REM Soft cap on the number of frames (keeps COLMAP time bounded; COLMAP costs ~1.5-2.5 s per frame).
set "MIN_FPS=3"            & REM Never go below this many frames/second, even if that passes MAX_FRAMES (fewer frames would not overlap enough for SfM).

REM ------------------------------------------- CAMERA POSES (COLMAP with GLOMAP-style global mapper) --
set "POSES=auto"           & REM auto = COLMAP first, DA3 as fallback if COLMAP fails its checks or is too slow ; colmap = COLMAP only (stop if it fails) ; da3 = skip COLMAP (fast, lower quality).
set "TIME_BUDGET_MIN=0"    & REM 0 = no limit. If >0 (e.g. 15) and POSES=auto: when the estimated total exceeds this many minutes, the faster DA3 route is used automatically.
set "SFM_MAPPER=global"    & REM global = GLOMAP-style global mapper built into COLMAP (fast) ; incremental = classic mapper (slower, more tolerant of weak footage).
set "SFM_FEATURES=3000"    & REM Max SIFT features per image. Lower (2000) = faster, higher (8000) = more matches on texture-poor scenes.
set "SFM_IMAGE_SIZE=1200"  & REM Longest image side used for SIFT, in pixels. Lower (800) = faster.
set "SFM_OVERLAP=8"        & REM Each frame is matched with this many following frames. Lower = faster; raise it for fast camera motion.
set "SFM_BA_ITERATIONS=1"  & REM Global mapper bundle-adjustment rounds. 1 is fast; 3 is slower and a little more accurate.
set "SFM_CAMERA_MODEL=SIMPLE_RADIAL"  & REM Lens model: SIMPLE_RADIAL (robust default), PINHOLE (no distortion), OPENCV (needs lots of well-spread views).
set "SFM_MIN_REGISTERED=0.8" & REM Fraction of frames COLMAP must register (0.8 = 80%) or the result is rejected.
set "FOCAL_FACTOR=0.72"    & REM Focal-length guess as a fraction of the longer image side. 0.7-0.75 = normal phone camera; ~0.5 = wide-angle; ~1.0 = zoomed in.
set "SFM_GPU=auto"         & REM auto = use GPU SIFT/matching when the COLMAP build has CUDA and a test succeeds (it crashes on this laptop, so CPU is used) ; off = never try.
set "COLMAP_EXE="          & REM Leave empty = tools\colmap-cpu\bin\colmap.exe. Or a full path to another colmap.exe (e.g. a working CUDA build on another PC).
set "SFM_TIMEOUT=1800"     & REM Seconds to wait for COLMAP before giving up (then DA3 is used when POSES=auto).

REM ------------------------------------------- RUN BEHAVIOUR ------------------------------------------
set "RESUME=0"             & REM 1 = continue an earlier run of the same NAME and skip stages whose results already exist (e.g. after an interruption). 0 = start fresh.
set "FORCE=0"              & REM 1 = reuse an existing runs\NAME folder without skipping anything. 0 = stop if it exists.
set "OPEN_VIEWER=1"        & REM 1 = open the finished splat in the Brush viewer automatically. 0 = do not.
set "PAUSE_AT_END=1"       & REM 1 = keep this window open at the end so you can read the summary. 0 = close it.
set "OUT_ROOT=runs"        & REM Folder (relative to this script, or a full path) where run folders are created.
set "BLENDER=C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"  & REM Blender executable (only needed for MODE=mesh or both).
set "PYTHON=python"        & REM Python executable (needs numpy, opencv, torch, scipy, pycolmap, transformers). Use a full path if "python" is not found.
REM ---------------------------------------------------------------------------------------------------------

echo.
echo ======================================================================
echo   VIDEO -^> 3D        mode: %MODE%
echo ======================================================================
echo   video       : %VIDEO%
echo   name        : %NAME%            output: %OUT_ROOT%
echo   frames      : fps %FPS%, cap %MAX_FRAMES%, never below %MIN_FPS% fps
echo   splat       : %STEPS% steps, %MAX_RES%px, up to %MAX_SPLATS% splats, held-out every %HELDOUT_EVERY%th frame
echo   poses       : %POSES%   mapper: %SFM_MAPPER%   features: %SFM_FEATURES%   image size: %SFM_IMAGE_SIZE%
echo                 overlap: %SFM_OVERLAP%   BA rounds: %SFM_BA_ITERATIONS%   camera: %SFM_CAMERA_MODEL%   focal guess: %FOCAL_FACTOR%
echo                 gpu: %SFM_GPU%   timeout: %SFM_TIMEOUT%s   time budget: %TIME_BUDGET_MIN% min   resume: %RESUME%
echo ======================================================================
echo   The percentage below is an ESTIMATE from typical stage times.
echo ======================================================================
echo.

if not exist "%VIDEO%" (
  echo ERROR: the video was not found: %VIDEO%
  echo Edit the VIDEO line at the top of this file.
  goto :finish
)
if not exist "scripts\run_splat.py" (
  echo ERROR: scripts\run_splat.py not found. Keep this file in the project root folder.
  goto :finish
)

set "EXTRA="
if defined NAME set EXTRA=!EXTRA! --name "!NAME!"
if defined COLMAP_EXE set EXTRA=!EXTRA! --colmap-exe "!COLMAP_EXE!"
if defined BLENDER set EXTRA=!EXTRA! --blender "!BLENDER!"
if "%FORCE%"=="1" set EXTRA=!EXTRA! --force
if "%RESUME%"=="1" set EXTRA=!EXTRA! --resume
if "%OPEN_VIEWER%"=="1" set EXTRA=!EXTRA! --open-viewer

"%PYTHON%" -u -I "scripts\run_splat.py" --video "%VIDEO%" --out-root "%OUT_ROOT%" --mode %MODE% ^
  --fps %FPS% --max-frames %MAX_FRAMES% --min-fps %MIN_FPS% ^
  --steps %STEPS% --max-res %MAX_RES% --max-splats %MAX_SPLATS% --heldout-every %HELDOUT_EVERY% ^
  --poses %POSES% --time-budget-min %TIME_BUDGET_MIN% --sfm-mapper %SFM_MAPPER% --sfm-features %SFM_FEATURES% ^
  --sfm-image-size %SFM_IMAGE_SIZE% --sfm-overlap %SFM_OVERLAP% --sfm-ba-iterations %SFM_BA_ITERATIONS% ^
  --sfm-camera-model %SFM_CAMERA_MODEL% --sfm-min-registered %SFM_MIN_REGISTERED% --focal-factor %FOCAL_FACTOR% ^
  --sfm-gpu %SFM_GPU% --sfm-timeout %SFM_TIMEOUT% !EXTRA!
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (echo FINISHED WITH AN ERROR ^(exit code %RC%^). Read the message above and the logs in %OUT_ROOT%\%NAME%\logs.) else (echo DONE. Results are in %OUT_ROOT%\%NAME%.)

:finish
if "%PAUSE_AT_END%"=="1" pause
endlocal
