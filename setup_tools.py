"""Download and install the third-party tools this project needs into the tools\\ folder (and the Python packages).

  python setup_tools.py --check                 show what is installed / missing, change nothing
  python setup_tools.py --dry-run               show exactly what would be downloaded, change nothing
  python setup_tools.py                         install everything that is missing (asks before downloading)
  python setup_tools.py --yes                   same, without the question
  python setup_tools.py --only brush,colmap     install just these components
  python setup_tools.py --skip deps             install everything except these
  python setup_tools.py --with-colmap-cuda      also install the COLMAP CUDA build (NOT used by default; see below)
  python setup_tools.py --force --only brush    reinstall one component (the old folder is renamed, never deleted)

Components (names for --only / --skip):
  deps         Python packages from requirements.txt (pip, uses this Python)
  brush        Brush v0.3.0 splat trainer/viewer             -> tools\\brush
  colmap       COLMAP 4.2.1, CPU-only Windows build          -> tools\\colmap-cpu   (the one the pipeline uses)
  colmap-cuda  COLMAP 4.2.1, CUDA Windows build (optional)   -> tools\\colmap-cuda  (its GPU path crashed on the dev laptop)
  da3-src      Depth Anything 3 source at a pinned commit    -> tools\\da3_src
  da3-venv     virtual environment + requirements-da3.txt    -> tools\\venv_da3
  da3-model    DA3-BASE weights (Apache-2.0)                 -> tools\\da3_models\\DA3-BASE
  dav2         Depth-Anything-V2 Small + Metric-Indoor-Small -> the Hugging Face cache (see note)

Not downloaded: Blender (install it yourself from https://www.blender.org/download/; only needed for MODE=mesh|both).

Safety: downloads come only from the URLs pinned below (GitHub releases / archive of a fixed commit / Hugging Face).
Archives with a known SHA-256 are verified before use; a mismatch aborts that component. The DA3 source archive of a
commit has no stable hash, so it is only checked structurally. Nothing is uploaded anywhere. Standard library only (the
Hugging Face downloads import huggingface_hub, which `deps` installs).

Note on the Depth-Anything-V2 models: the scripts load them by name from the standard Hugging Face cache
(%USERPROFILE%\\.cache\\huggingface\\hub, or HF_HUB_CACHE / HF_HOME if set), offline. This script therefore writes them
there, with real file copies (no symlinks, which need a privilege this Windows account lacks).

Tested: --check and --dry-run and the helper functions (archive layout detection, SHA check, cache layout). A real
download/install on a clean machine has NOT been run yet.
"""
import argparse
import hashlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
TMP = TOOLS / "_setup_tmp"          # scratch space (same drive as the targets so moves are renames); removed afterwards
REQ = ROOT / "requirements.txt"
REQ_DA3 = ROOT / "requirements-da3.txt"
BLENDER_DEFAULT = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")

GH = "https://github.com"
ARCHIVES = {
    "brush": dict(
        title="Brush v0.3.0 (Apache-2.0)",
        url=f"{GH}/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-pc-windows-msvc.zip",
        sha256="b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6", size_mb=152,
        target="brush", exe="brush_app.exe", root_is_exe_parent=True),
    "colmap": dict(
        title="COLMAP 4.2.1 CPU-only (BSD-style)",
        url=f"{GH}/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-nocuda.zip",
        sha256="c493d88cb4a43f21afba42cb55bf786e5434a0420e73db0ba1454c5d8b3686ae", size_mb=122,
        target="colmap-cpu", exe="colmap.exe", root_is_exe_parent=False),
    "colmap-cuda": dict(
        title="COLMAP 4.2.1 CUDA (optional, not used by default)",
        url=f"{GH}/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip",
        sha256="e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881", size_mb=396,
        target="colmap-cuda", exe="colmap.exe", root_is_exe_parent=False),
}
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_SRC_URL = f"{GH}/ByteDance-Seed/Depth-Anything-3/archive/{DA3_COMMIT}.zip"
DA3_MODEL = dict(repo="depth-anything/DA3-BASE", patterns=["config.json", "model.safetensors", "*.md"],
                 sha256="e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5", size_mb=520)
DAV2_MODELS = [
    dict(repo="depth-anything/Depth-Anything-V2-Small-hf", patterns=["config.json", "model.safetensors", "preprocessor_config.json"], size_mb=100),
    dict(repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", patterns=["config.json", "model.safetensors", "preprocessor_config.json"], size_mb=100),
]
ORDER = ["deps", "brush", "colmap", "colmap-cuda", "da3-src", "da3-venv", "da3-model", "dav2"]
DEFAULT_OFF = {"colmap-cuda"}
APPROX_MB = {"deps": 4000, "brush": 152, "colmap": 122, "colmap-cuda": 396, "da3-src": 50, "da3-venv": 100,
             "da3-model": 520, "dav2": 200}   # download sizes; installed size is larger (COLMAP about 340 MB / 760 MB unpacked)


# ---------------------------------------------------------------------------------------------------- helpers
def say(msg=""):
    print(msg, flush=True)


def sha256_of(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest, expected_sha=None):
    """Download to <dest>.part, verify SHA-256 if given, then rename. Raises on any problem."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "video-to-3d-setup"})
    t0 = last = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 5:
                last = time.time()
                pct = f"{100 * got / total:.0f}%" if total else ""
                say(f"    {got / 2**20:7.1f} MB {pct}  ({got / 2**20 / max(time.time() - t0, 1e-6):.1f} MB/s)")
    if expected_sha:
        got_sha = sha256_of(part)
        if got_sha.lower() != expected_sha.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {url}\n  expected {expected_sha}\n  got      {got_sha}\n  (file deleted; nothing installed)")
        say("    SHA-256 verified")
    else:
        say("    (no pinned hash for this download; structure is checked instead)")
    part.replace(dest)
    return dest


def safe_extract(zip_path, dest):
    """Extract a zip, refusing members that would land outside `dest` (zip-slip)."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            out = (dest / m.filename).resolve()
            if dest != out and dest not in out.parents:
                raise RuntimeError(f"unsafe path in archive: {m.filename}")
        z.extractall(dest)


def find_exe(tree, exe_name):
    """Locate exe_name in an extracted tree; prefer the shallowest match whose parent is 'bin' (COLMAP) else shallowest."""
    hits = sorted(Path(tree).rglob(exe_name), key=lambda p: len(p.parts))
    if not hits:
        raise RuntimeError(f"{exe_name} not found in the extracted archive")
    bins = [p for p in hits if p.parent.name.lower() == "bin"]
    return (bins or hits)[0]


def single_top_dir(tree):
    items = [p for p in Path(tree).iterdir()]
    return items[0] if len(items) == 1 and items[0].is_dir() else Path(tree)


def move_into_place(src, target, force):
    """Move `src` to `target`. An existing target is renamed (never deleted) and reported."""
    target = Path(target)
    if target.exists():
        if not force:
            raise RuntimeError(f"{target} already exists (use --force to replace it; the old one is renamed, not deleted)")
        old = target.with_name(target.name + ".bak_" + time.strftime("%Y%m%d_%H%M%S"))
        target.rename(old)
        say(f"    existing folder renamed to {old.name} (delete it yourself when you no longer need it)")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(target))


def run(cmd, **kw):
    say("    $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def req_pins(path):
    """{normalised name: version} for lines like name==version (ignores comments, options, unpinned lines)."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#;]+)", line)
        if m:
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def installed_version(name, python=None):
    """Version of a package in `python` (default: this interpreter), or None."""
    if python is None or Path(python).resolve() == Path(sys.executable).resolve():
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None
    r = subprocess.run([str(python), "-I", "-c", f"import importlib.metadata as m;print(m.version({name!r}))"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def hf_cache_dir():
    for var in ("HF_HUB_CACHE",):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_has(repo, cache=None, needs=("config.json", "model.safetensors")):
    """True if refs/main of the repo resolves to a snapshot folder holding the needed files."""
    base = Path(cache or hf_cache_dir()) / ("models--" + repo.replace("/", "--"))
    ref = base / "refs" / "main"
    if not ref.is_file():
        return False
    snap = base / "snapshots" / ref.read_text().strip()
    return all((snap / n).is_file() for n in needs)


def install_into_hf_cache(repo, src_dir, sha, cache=None):
    """Copy files (real copies, no symlinks) into the HF cache layout so offline from_pretrained(repo) works."""
    base = Path(cache or hf_cache_dir()) / ("models--" + repo.replace("/", "--"))
    snap = base / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    for p in Path(src_dir).iterdir():
        if p.is_file():
            shutil.copy2(p, snap / p.name)
    (base / "refs").mkdir(parents=True, exist_ok=True)
    (base / "refs" / "main").write_text(sha)
    return snap


def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


# ------------------------------------------------------------------------------------------- component status
def status(key, python):
    """(ok, detail) for one component; never changes anything."""
    if key == "deps":
        pins = req_pins(REQ)
        bad = []
        for name, want in pins.items():
            have = installed_version(name, python)
            if have is None:
                bad.append(f"{name} missing")
            elif have.split("+")[0] != want.split("+")[0]:
                bad.append(f"{name} {have} (tested {want})")
        return (not bad, "all pinned packages present" if not bad else "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))
    if key in ARCHIVES:
        a = ARCHIVES[key]
        exe = TOOLS / a["target"] / ("bin" if not a["root_is_exe_parent"] else "") / a["exe"]
        return (exe.is_file(), str(exe.relative_to(ROOT)) if exe.is_file() else f"missing: {exe.relative_to(ROOT)}")
    if key == "da3-src":
        ok = (TOOLS / "da3_src" / "src" / "depth_anything_3").is_dir()
        return (ok, "tools\\da3_src\\src\\depth_anything_3 present" if ok else "missing: tools\\da3_src\\src\\depth_anything_3")
    if key == "da3-venv":
        py = TOOLS / "venv_da3" / "Scripts" / "python.exe"
        if not py.is_file():
            return (False, "missing: tools\\venv_da3")
        bad = [n for n in req_pins(REQ_DA3) if installed_version(n, py) is None]
        return (not bad, "venv with DA3 extras present" if not bad else "venv exists, packages missing: " + ", ".join(bad))
    if key == "da3-model":
        f = TOOLS / "da3_models" / "DA3-BASE" / "model.safetensors"
        return (f.is_file(), "tools\\da3_models\\DA3-BASE\\model.safetensors present" if f.is_file() else "missing: tools\\da3_models\\DA3-BASE\\model.safetensors")
    if key == "dav2":
        miss = [m["repo"] for m in DAV2_MODELS if not hf_cache_has(m["repo"], needs=("config.json", "model.safetensors", "preprocessor_config.json"))]
        return (not miss, f"in HF cache ({hf_cache_dir()})" if not miss else "missing in HF cache: " + ", ".join(miss))
    raise KeyError(key)


# ------------------------------------------------------------------------------------------- component installers
def install_archive(key, force, dry):
    a = ARCHIVES[key]
    say(f"  {a['title']}: {a['url']}")
    if dry:
        return
    TMP.mkdir(parents=True, exist_ok=True)
    zpath = download(a["url"], TMP / (key + ".zip"), a["sha256"])
    ex = TMP / ("x_" + key)
    shutil.rmtree(ex, ignore_errors=True)
    say("    extracting ...")
    safe_extract(zpath, ex)
    exe = find_exe(ex, a["exe"])
    root = exe.parent if a["root_is_exe_parent"] else exe.parent.parent
    move_into_place(root, TOOLS / a["target"], force)
    zpath.unlink(missing_ok=True)
    shutil.rmtree(ex, ignore_errors=True)
    # smoke test
    exe_final = TOOLS / a["target"] / ("bin" if not a["root_is_exe_parent"] else "") / a["exe"]
    try:
        r = subprocess.run([str(exe_final), "--version" if key == "brush" else "help"], capture_output=True, text=True, timeout=30)
        out = (r.stdout or "") + (r.stderr or "")
        want = "0.3.0" if key == "brush" else "4.2.1"
        say(f"    version check: {'OK' if want in out else 'WARNING: expected ' + want + ' not found in the output'}")
    except Exception as e:
        say(f"    WARNING: could not run {exe_final.name}: {e}")


def install_deps(python, dry):
    say(f"  pip install -r requirements.txt   (python: {python})")
    if dry:
        return
    r = run([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", REQ])
    if r.returncode != 0:
        raise RuntimeError("pip install failed (see the output above)")


def install_da3_src(force, dry):
    say(f"  Depth Anything 3 source, commit {DA3_COMMIT[:8]}: {DA3_SRC_URL}")
    if dry:
        return
    TMP.mkdir(parents=True, exist_ok=True)
    zpath = download(DA3_SRC_URL, TMP / "da3_src.zip", None)
    ex = TMP / "x_da3"
    shutil.rmtree(ex, ignore_errors=True)
    safe_extract(zpath, ex)
    top = single_top_dir(ex)
    if not (top / "src" / "depth_anything_3").is_dir():
        raise RuntimeError("unexpected archive layout: src/depth_anything_3 not found")
    move_into_place(top, TOOLS / "da3_src", force)
    zpath.unlink(missing_ok=True)
    shutil.rmtree(ex, ignore_errors=True)


def install_da3_venv(python, force, dry):
    venv = TOOLS / "venv_da3"
    vpy = venv / "Scripts" / "python.exe"
    say(f"  venv {venv.relative_to(ROOT)} (--system-site-packages) + requirements-da3.txt with numpy/torch pinned")
    if dry:
        return
    if vpy.is_file() and force:
        old = venv.with_name(venv.name + ".bak_" + time.strftime("%Y%m%d_%H%M%S"))
        venv.rename(old)
        say(f"    existing venv renamed to {old.name}")
    if not vpy.is_file():
        r = run([python, "-m", "venv", "--system-site-packages", venv])
        if r.returncode != 0 or not vpy.is_file():
            raise RuntimeError("could not create the virtual environment")
    out = subprocess.run([str(vpy), "-I", "-c", "import numpy, torch; print('numpy==' + numpy.__version__); print('torch==' + torch.__version__)"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("numpy/torch are not importable from the venv: run the 'deps' step first")
    TMP.mkdir(parents=True, exist_ok=True)
    cons = TMP / "constraints.txt"
    cons.write_text(out.stdout, encoding="ascii")
    r = run([vpy, "-m", "pip", "install", "--disable-pip-version-check", "-c", cons, "-r", REQ_DA3])
    if r.returncode != 0:
        raise RuntimeError("pip install into the DA3 venv failed")


def hf_download(repo, patterns, dest):
    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub is not installed: run the 'deps' step first")
    sha = HfApi().model_info(repo).sha
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=patterns)
    return sha


def install_da3_model(force, dry):
    say(f"  {DA3_MODEL['repo']} (Apache-2.0) -> tools\\da3_models\\DA3-BASE  (about {DA3_MODEL['size_mb']} MB)")
    if dry:
        return
    dest = TOOLS / "da3_models" / "DA3-BASE"
    if dest.exists() and force:
        old = dest.with_name(dest.name + ".bak_" + time.strftime("%Y%m%d_%H%M%S"))
        dest.rename(old)
        say(f"    existing folder renamed to {old.name}")
    # local_dir download: avoids the Hugging Face cache symlinks (WinError 1314 on accounts without the privilege)
    hf_download(DA3_MODEL["repo"], DA3_MODEL["patterns"], dest)
    f = dest / "model.safetensors"
    if not f.is_file():
        raise RuntimeError("model.safetensors was not downloaded")
    got = sha256_of(f)
    if got.lower() != DA3_MODEL["sha256"]:
        raise RuntimeError(f"SHA-256 of model.safetensors differs from the recorded one:\n  expected {DA3_MODEL['sha256']}\n  got      {got}\n"
                           "  The file was kept for inspection; the Hugging Face repo may have been updated. Ask the project owner before using it.")
    say("    SHA-256 verified")


def install_dav2(force, dry):
    for m in DAV2_MODELS:
        say(f"  {m['repo']} -> Hugging Face cache {hf_cache_dir()}  (about {m['size_mb']} MB)")
    if dry:
        return
    for m in DAV2_MODELS:
        needs = ("config.json", "model.safetensors", "preprocessor_config.json")
        if hf_cache_has(m["repo"], needs=needs) and not force:
            say(f"    {m['repo']}: already in the cache")
            continue
        tmp = Path(tempfile.mkdtemp(prefix="dav2_", dir=TOOLS))
        try:
            sha = hf_download(m["repo"], m["patterns"], tmp)
            if not all((tmp / n).is_file() for n in needs):
                raise RuntimeError(f"{m['repo']}: expected files missing after download")
            snap = install_into_hf_cache(m["repo"], tmp, sha)
            say(f"    installed into {snap}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="show the status of every component and exit")
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded/installed and exit")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--only", default="", help="comma-separated components: " + ", ".join(ORDER))
    ap.add_argument("--skip", default="", help="comma-separated components to skip")
    ap.add_argument("--with-colmap-cuda", action="store_true", help="also install the (optional) COLMAP CUDA build")
    ap.add_argument("--force", action="store_true", help="reinstall selected components (old folders are renamed, not deleted)")
    ap.add_argument("--python", default=sys.executable, help="Python used for the 'deps' step and for creating the DA3 venv (default: this one)")
    a = ap.parse_args()

    if os.name != "nt":
        say("WARNING: this project's tools are Windows builds (Brush/COLMAP zips); on another OS install them yourself.")
    if sys.version_info < (3, 10):
        say(f"WARNING: Python {sys.version.split()[0]} is older than anything tested (3.13.7).")
    elif sys.version_info[:2] != (3, 13):
        say(f"NOTE: tested with Python 3.13.7; you have {sys.version.split()[0]}. The pinned wheels may not exist for your version.")

    for name in (x for x in (a.only + "," + a.skip).split(",") if x.strip()):
        if name.strip() not in ORDER:
            raise SystemExit(f"unknown component '{name}'. Choose from: {', '.join(ORDER)}")
    selected = [k for k in ORDER if (not a.only or k in {x.strip() for x in a.only.split(",")})
                and k not in {x.strip() for x in a.skip.split(",") if x.strip()}]
    if not a.only and not a.with_colmap_cuda:
        selected = [k for k in selected if k not in DEFAULT_OFF]

    say(f"Project: {ROOT}\nTools folder: {TOOLS}\n")
    st = {k: status(k, a.python) for k in ORDER}
    say("Status:")
    for k in ORDER:
        ok, detail = st[k]
        say(f"  [{'OK     ' if ok else 'MISSING'}] {k:12s} {detail}")
    blender = BLENDER_DEFAULT
    say(f"  [{'OK     ' if blender.is_file() else 'MISSING'}] {'blender':12s} {blender if blender.is_file() else 'not installed at ' + str(blender) + ' (only needed for MODE=mesh|both; install from https://www.blender.org/download/)'}")
    if a.check:
        return 0

    todo = [k for k in selected if a.force or not st[k][0]]
    if not todo:
        say("\nNothing to do: everything selected is already installed.")
        return 0
    need_mb = sum(APPROX_MB[k] for k in todo)
    free = free_gb(TOOLS)
    say(f"\nWill install: {', '.join(todo)}   (download about {need_mb / 1024:.1f} GB; free disk {free:.1f} GB)")
    if free * 1024 < need_mb * 1.6 + 1024:
        say("WARNING: free disk looks too small for this (installed size is larger than the download; archives are unpacked next to the targets).")
    for k in todo:
        say(f"- {k}")
        if k == "deps":
            say("  Python packages from requirements.txt (PyTorch CUDA wheels alone are several GB)")
        elif k in ARCHIVES:
            say(f"  {ARCHIVES[k]['url']}")
        elif k == "da3-src":
            say(f"  {DA3_SRC_URL}")
        elif k == "da3-model":
            say(f"  Hugging Face: {DA3_MODEL['repo']}")
        elif k == "dav2":
            for m in DAV2_MODELS:
                say(f"  Hugging Face: {m['repo']}")
        elif k == "da3-venv":
            say("  local virtual environment + packages from requirements-da3.txt (PyPI)")
    if a.dry_run:
        say("\n--dry-run: nothing was downloaded or changed.")
        return 0
    if not a.yes:
        if input("\nProceed with these downloads? [y/N] ").strip().lower() not in ("y", "yes"):
            say("Cancelled. Nothing was changed.")
            return 1

    failures = []
    for k in todo:
        say(f"\n== {k}")
        try:
            if k == "deps":
                install_deps(a.python, False)
            elif k in ARCHIVES:
                install_archive(k, a.force, False)
            elif k == "da3-src":
                install_da3_src(a.force, False)
            elif k == "da3-venv":
                install_da3_venv(a.python, a.force, False)
            elif k == "da3-model":
                install_da3_model(a.force, False)
            elif k == "dav2":
                install_dav2(a.force, False)
        except Exception as e:  # report and carry on with the independent components
            failures.append((k, str(e)))
            say(f"  FAILED: {e}")
    shutil.rmtree(TMP, ignore_errors=True)

    say("\n== final status")
    st = {k: status(k, a.python) for k in ORDER}
    for k in ORDER:
        say(f"  [{'OK     ' if st[k][0] else 'MISSING'}] {k:12s} {st[k][1]}")
    if failures:
        say("\nFailed components: " + ", ".join(k for k, _ in failures) + "\nFix the cause (see messages above) and run this script again; finished components are skipped.")
        return 1
    say("\nDone. Next: python -I scripts\\run_splat.py --video video\\<your video>.mp4 --plan-only   (or edit and run run_video_to_splat.bat)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
