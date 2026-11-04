"""
Fetch, verify and import the upstream CorridorKey 2.0 runtime.

This pack ships none of Corridor Digital's code. On first use it downloads the
pinned upstream commit from GitHub into ``<pack>/upstream/`` and checks every
file against ``UPSTREAM_MANIFEST.json`` (SHA-256 per file) before anything is
imported. Only the files the manifest names are extracted; nothing else in the
archive is ever written to disk.

Why fetch instead of vendoring: the Corridor Key Licence (section 2a) forbids
redistributing the tool "in whole or in part ... as part of a competing
product". Keeping their code out of this repository keeps us clear of that
clause; this pack only drives it.

The upstream package is imported under a private name
(``_bat_corridorkey2_upstream.corridorkey``) so it cannot collide with the
other ComfyUI CorridorKey packs, which put upstream on sys.path as a top-level
``corridorkey`` package.
"""

import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import types
import urllib.request

logger = logging.getLogger("Bat_CorridorKey2")

_HERE = os.path.dirname(os.path.abspath(__file__))
PACK_ROOT = os.path.dirname(_HERE)

with open(os.path.join(_HERE, "UPSTREAM_MANIFEST.json"), encoding="utf-8") as _f:
    MANIFEST = json.load(_f)

REPO = MANIFEST["repo"]
COMMIT = MANIFEST["commit"]
FILES = MANIFEST["files"]          # {relpath: sha256}

UPSTREAM_DIR = os.path.join(PACK_ROOT, "upstream", f"CorridorKey2.0-{COMMIT[:12]}")
TARBALL_URL = f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}"

_PKG = "_bat_corridorkey2_upstream"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root=UPSTREAM_DIR):
    """Return the list of manifest files that are missing or altered."""
    bad = []
    for rel, digest in FILES.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p) or _sha256(p) != digest:
            bad.append(rel)
    return bad


def _download_tarball():
    logger.info("fetching CorridorKey 2.0 runtime %s@%s", REPO, COMMIT[:12])
    req = urllib.request.Request(TARBALL_URL, headers={"User-Agent": "ComfyUI-BAT-CorridorKey2"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _extract(blob, dest):
    """Write exactly the manifest's files from the archive into ``dest``.

    Members are matched by name against the manifest, never written by the
    archive's own path, so a hostile tarball cannot traverse out of ``dest``.
    """
    wanted = set(FILES)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # GitHub archives wrap everything in "<repo>-<sha>/".
            parts = member.name.split("/", 1)
            if len(parts) != 2 or parts[1] not in wanted:
                continue
            rel = parts[1]
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != FILES[rel]:
                raise RuntimeError(
                    f"CorridorKey upstream file {rel} does not match the pinned "
                    f"manifest; refusing to install it.")
            out = os.path.join(dest, *rel.split("/"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(data)
            wanted.discard(rel)
    if wanted:
        raise RuntimeError(
            "CorridorKey upstream archive is missing pinned files: "
            + ", ".join(sorted(wanted)[:8]))


def ensure_upstream(allow_download=True):
    """Make sure the verified upstream tree exists, fetching it if allowed."""
    if not verify():
        return UPSTREAM_DIR
    if not allow_download:
        raise RuntimeError(
            f"The CorridorKey 2.0 runtime is not installed at {UPSTREAM_DIR} and "
            f"auto_download is off. Turn auto_download on, or extract "
            f"{TARBALL_URL} there by hand.")
    parent = os.path.dirname(UPSTREAM_DIR)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".ck2_fetch_", dir=parent)
    try:
        _extract(_download_tarball(), staging)
        if os.path.isdir(UPSTREAM_DIR):
            shutil.rmtree(UPSTREAM_DIR)
        os.replace(staging, UPSTREAM_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    bad = verify()
    if bad:
        raise RuntimeError(f"CorridorKey upstream install failed verification: {bad[:8]}")
    logger.info("CorridorKey 2.0 runtime installed at %s", UPSTREAM_DIR)
    return UPSTREAM_DIR


@contextlib.contextmanager
def isolated_build_env():
    """Context for constructing the model.

    Two things upstream does process-wide that must not leak into ComfyUI:

    * ``prepare_runtime_environment`` sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
      for good, which would silently stop every other pack's auto-download for
      the rest of the session. We set them only while building.
    * MoGe is imported as a top-level ``moge`` package. If another pack already
      imported a different ``moge``, upstream would get that one. Any foreign
      ``moge`` modules are set aside while ours is imported, then put back.
    """
    moge_root = os.path.join(UPSTREAM_DIR, "third_party", "MoGe")
    saved_env = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    foreign = {k: v for k, v in sys.modules.items()
               if (k == "moge" or k.startswith("moge."))
               and not str(getattr(v, "__file__", "") or "").startswith(moge_root)}
    for k in foreign:
        del sys.modules[k]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, moge_root)
    try:
        yield
    finally:
        try:
            sys.path.remove(moge_root)
        except ValueError:
            pass
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Drop our moge from sys.modules either way: the built model keeps its
        # own references, and a later ``import moge`` elsewhere must not get it.
        for k in [k for k in sys.modules if k == "moge" or k.startswith("moge.")]:
            del sys.modules[k]
        sys.modules.update(foreign)


def load_package(allow_download=True):
    """Import and return the upstream ``corridorkey`` package (private name)."""
    full = _PKG + ".corridorkey"
    if full in sys.modules:
        return sys.modules[full]
    root = ensure_upstream(allow_download=allow_download)
    pkg = sys.modules.get(_PKG)
    if pkg is None:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [root]
        pkg.__file__ = os.path.join(root, "__init__.py")
        sys.modules[_PKG] = pkg
    return importlib.import_module(full)


def submodule(name, allow_download=True):
    """``submodule("runtime")`` -> the upstream ``corridorkey.runtime`` module."""
    load_package(allow_download=allow_download)
    return importlib.import_module(f"{_PKG}.corridorkey.{name}")


def load_script(relpath, allow_download=True):
    """Import an upstream script (e.g. scripts/download_weights.py) by path."""
    root = ensure_upstream(allow_download=allow_download)
    name = f"{_PKG}_script_" + relpath.replace("/", "_").replace(".py", "")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, *relpath.split("/")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
