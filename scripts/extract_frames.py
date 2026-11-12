"""Extract sharp, evenly spaced frames from a video for photogrammetry.

Usage:
    python -I scripts/extract_frames.py <video> <out_dir> [--window 8] [--max-side 1600] [--target-max 400] [--dry-run]

Picks the sharpest frame (variance of Laplacian) in each window of N frames.
Streams the video, so memory stays small even for 4K input. --max-side downscales
frames on save, which cuts disk use and SfM/MVS time (1600 px is a good low-resource default).
"""
import argparse
import os
import sys

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--window", type=int, default=8,
                    help="frames per window; 8 @ 30 fps is about 3.75 kept frames/s")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="kept frames per second of video (general: independent of the source frame rate); "
                         "overrides --window. ~6-10 is good for a walking pace.")
    ap.add_argument("--target-frames", type=int, default=0,
                    help="frame budget: overrides --window so a long video never yields more than ~N frames "
                         "(keeps SfM time roughly flat; ~2 kept frames/s is a good walking pace)")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="downscale so the longer side is at most this many pixels (0 = keep original)")
    ap.add_argument("--target-max", type=int, default=400,
                    help="warn if more frames than this would be saved")
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many frames would be saved without writing")
    args = ap.parse_args()

    if args.window < 1:
        print("--window must be >= 1", file=sys.stderr)
        return 2
    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.fps > 0 and fps > 0:
        args.window = max(1, int(round(fps / args.fps)))
    if args.target_frames and total > 0:
        args.window = max(args.window if args.fps > 0 else 1, -(-total // args.target_frames))  # budget caps the count
    expected = (total + args.window - 1) // args.window if total > 0 else -1
    print(f"Video: {total} frames @ {fps:.2f} fps; window={args.window}; expect ~{expected} output frames")
    if expected > args.target_max:
        print(f"WARNING: above {args.target_max}. Increase --window.")
    if args.dry_run:
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.out_dir) if f.lower().endswith(".jpg")]
    if existing:
        print(f"Refusing to write: {args.out_dir} already has {len(existing)} jpg files. Use an empty folder.",
              file=sys.stderr)
        return 2

    best_score, best_frame = -1.0, None
    count_in_window, saved = 0, 0

    def flush():
        nonlocal best_score, best_frame, count_in_window, saved
        if best_frame is not None:
            img = best_frame
            h, w = img.shape[:2]
            if args.max_side and max(h, w) > args.max_side:
                s = args.max_side / max(h, w)
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            path = os.path.join(args.out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        best_score, best_frame, count_in_window = -1.0, None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score, best_frame = score, frame  # keep only the current best, not the whole window
        count_in_window += 1
        if count_in_window == args.window:
            flush()
    flush()
    cap.release()
    print(f"Saved {saved} frames to {args.out_dir}")
    import json  # downstream stages use this to express neighbour/loop spans in seconds, not frame counts
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(dict(source=os.path.basename(args.video), source_fps=fps, window=args.window,
                       kept_fps=(fps / args.window if fps > 0 else None), frames=saved, source_frames=total), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
