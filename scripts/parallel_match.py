"""Sequential-style feature matching split across several COLMAP processes (CPU builds).

WHY: measured on this laptop (22 logical CPUs), COLMAP's CPU `sequential_matcher` keeps the machine at ~10% CPU
(about 2 cores) whatever --FeatureMatching.num_threads is, and matching was 92 of 171 s of the mouse-clip SfM stage.
This script generates the same pair set as sequential matching (frame i with i+1..i+overlap, plus the quadratic
offsets 2^k), deals the pairs to N workers, runs `colmap matches_importer` on N private copies of the database, and merges
the `matches` and `two_view_geometries` rows back into the real database. Pair sets are disjoint, so the merge is exact.

  python -I scripts/parallel_match.py --colmap-exe <colmap.exe> --database colmap/database.db --overlap 8 [--workers 0]

Workers are capped by free disk space (each worker needs a copy of the database: ~0.7 MB per frame). With one worker
(or too little disk) the plain sequential_matcher is used. Extra COLMAP flags can be passed after `--`.
"""
import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time


def ordered_names(db):
    return [r[0] for r in db.execute("SELECT name FROM images ORDER BY name")]


def sequential_pairs(n, overlap, quadratic=True):
    """Same pair rule as COLMAP's sequential matcher with loop detection off (image order = sorted names)."""
    offs = set(range(1, overlap + 1))
    if quadratic:
        offs |= {1 << i for i in range(overlap)}
    return sorted((i, i + o) for i in range(n) for o in offs if i + o < n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap-exe", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--no-quadratic", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (about a third of the logical CPUs, at most 8)")
    ap.add_argument("--tmp-dir", default="", help="where the worker databases go (default: next to the database)")
    ap.add_argument("extra", nargs="*", help="extra COLMAP flags for matches_importer (after --)")
    a = ap.parse_args()

    db_path = os.path.abspath(a.database)
    work_dir = os.path.abspath(a.tmp_dir) if a.tmp_dir else os.path.dirname(db_path)
    db = sqlite3.connect(db_path)
    names = ordered_names(db)
    pairs = sequential_pairs(len(names), a.overlap, not a.no_quadratic)
    size = os.path.getsize(db_path)
    free = shutil.disk_usage(work_dir).free
    cpu = os.cpu_count() or 4
    want = a.workers if a.workers > 0 else min(8, max(1, cpu // 3))
    fit = int(max(0, free - 1.0 * 2 ** 30) // max(size, 1))      # keep 1 GB of disk free
    workers = max(1, min(want, fit, len(pairs) // 40 or 1))
    print(f"parallel matching: {len(names)} images, {len(pairs)} pairs, {workers} worker(s) "
          f"(wanted {want}, disk allows {fit}, database {size / 2 ** 20:.0f} MB)", flush=True)
    extra = list(a.extra)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    t0 = time.time()
    if workers == 1:
        db.close()
        cmd = [a.colmap_exe, "sequential_matcher", "--database_path", db_path, "--SequentialMatching.overlap", str(a.overlap),
               "--SequentialMatching.loop_detection", "0"] + extra
        return subprocess.run(cmd).returncode

    tmp = tempfile.mkdtemp(prefix="pmatch_", dir=work_dir)
    procs = []
    try:
        shards = [pairs[k::workers] for k in range(workers)]       # interleaved: every worker gets a similar mix of near/far pairs
        for k, shard in enumerate(shards):
            wdb = os.path.join(tmp, f"w{k}.db")
            shutil.copyfile(db_path, wdb)
            lst = os.path.join(tmp, f"pairs{k}.txt")
            with open(lst, "w") as f:
                f.writelines(f"{names[i]} {names[j]}\n" for i, j in shard)
            log = open(os.path.join(tmp, f"w{k}.log"), "w")
            cmd = [a.colmap_exe, "matches_importer", "--database_path", wdb, "--match_list_path", lst, "--match_type", "pairs"] + extra
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, wdb, k))
        bad = []
        for p, log, wdb, k in procs:
            rc = p.wait()
            log.close()
            if rc != 0:
                bad.append((k, rc))
        if bad:
            for k, rc in bad:
                print(f"worker {k} failed with exit code {rc}; log tail:", flush=True)
                print(open(os.path.join(tmp, f"w{k}.log"), errors="replace").read()[-600:], flush=True)
            return 1
        print(f"workers finished in {time.time() - t0:.0f} s; merging", flush=True)
        for table in ("matches", "two_view_geometries"):
            cols = [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
            cl = ", ".join(cols)
            for p, log, wdb, k in procs:
                db.execute("ATTACH DATABASE ? AS w", (wdb,))
                wcols = {r[1] for r in db.execute(f"PRAGMA w.table_info({table})")}
                if not set(cols) <= wcols:
                    raise SystemExit(f"schema mismatch in {table}")
                db.execute(f"INSERT OR REPLACE INTO main.{table} ({cl}) SELECT {cl} FROM w.{table}")
                db.commit()
                db.execute("DETACH DATABASE w")
        n_tvg = db.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        print(f"merged: {n_tvg} verified pairs in {time.time() - t0:.0f} s total", flush=True)
        db.close()
        return 0
    finally:
        try:
            db.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
