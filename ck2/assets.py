"""
Weights: where they live, how they are fetched, and how the model is built.

Layout under ``ComfyUI/models/corridorkey/`` mirrors upstream's own
``weights/`` + ``vfms/`` folders, so upstream's ``download_weights.py`` can be
pointed straight at it:

    corridorkey/
      weights/corridorkey_v2.safetensors      CorridorKey v2 (2.5 GB)
      weights/foundation_stats.safetensors
      vfms/cradio/C-RADIOv4-SO400M/           NVIDIA C-RADIOv4 snapshot (1.7 GB)
      vfms/moge/model.pt                      Microsoft MoGe-2 ViT-B (400 MB)
      vfms/rvm/pretrain_rvm_small16_...npz    DeepMind RVM small (258 MB)

Every file comes from its original publisher at the revision upstream pins;
nothing is re-hosted.
"""

import logging
import os
import threading

import folder_paths
import torch

from . import upstream

logger = logging.getLogger("Bat_CorridorKey2")

FOLDER_KEY = "corridorkey"
MODELS_DIR = os.path.join(folder_paths.models_dir, FOLDER_KEY)

try:
    folder_paths.add_model_folder_path(FOLDER_KEY, MODELS_DIR)
except Exception:
    pass

RVM_FILE = "pretrain_rvm_small16_256_204031069.npz"


def asset_paths(root=MODELS_DIR):
    return {
        "weights": os.path.join(root, "weights", "corridorkey_v2.safetensors"),
        "foundation_stats": os.path.join(root, "weights", "foundation_stats.safetensors"),
        "cradio_dir": os.path.join(root, "vfms", "cradio", "C-RADIOv4-SO400M"),
        "moge_checkpoint": os.path.join(root, "vfms", "moge", "model.pt"),
        "rvm_ckpt": os.path.join(root, "vfms", "rvm", RVM_FILE),
    }


def missing_assets(root=MODELS_DIR):
    paths = asset_paths(root)
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if "cradio_dir" not in missing and not os.path.isfile(
            os.path.join(paths["cradio_dir"], "model.safetensors")):
        missing.append("cradio_dir")
    return missing


def ensure_weights(allow_download=True, root=MODELS_DIR):
    missing = missing_assets(root)
    if not missing:
        return asset_paths(root)
    if not allow_download:
        raise RuntimeError(
            f"CorridorKey weights missing in {root}: {', '.join(missing)}. Turn "
            f"auto_download on, or run upstream's scripts/download_weights.py "
            f"--package_root {root}")
    dl = upstream.load_script("scripts/download_weights.py")
    from pathlib import Path
    target = Path(root)
    logger.info("downloading CorridorKey 2.0 weights (~5 GB) into %s", root)
    if any(k in missing for k in ("weights", "foundation_stats")):
        dl.download_corridorkey(target, dl.CORRIDORKEY_REPO, None)
    if "cradio_dir" in missing:
        dl.download_cradio(target)
    if "moge_checkpoint" in missing:
        dl.download_moge(target)
    if "rvm_ckpt" in missing:
        dl.download_rvm(target)
    still = missing_assets(root)
    if still:
        raise RuntimeError(f"CorridorKey weights still missing after download: {still}")
    return asset_paths(root)


# One model per weight set, shared by every key node in the graph. The 1.27B
# parameters cost ~5 GB of RAM, so nothing is loaded twice.
_CACHE = {}
_LOCK = threading.Lock()


def _cache_key(paths):
    p = paths["weights"]
    return (os.path.realpath(p), os.path.getmtime(p))


def load_model(allow_download=True):
    """Build CorridorKey v2 on the CPU. Stages move to the GPU only while keying."""
    paths = ensure_weights(allow_download=allow_download)
    key = _cache_key(paths)
    with _LOCK:
        model = _CACHE.get(key)
        if model is not None:
            return model
        _CACHE.clear()
        runtime = upstream.submodule("runtime", allow_download=allow_download)
        model_mod = upstream.submodule("model")
        ckpt = runtime.load_runtime_checkpoint(paths["weights"])
        if ckpt.get("format") != "corridorkey_v2_runtime_checkpoint":
            raise RuntimeError(f"{paths['weights']} is not a CorridorKey v2 runtime checkpoint")
        # Same overrides upstream's runtime.load_model applies, except device:
        # upstream builds the foundations straight onto CUDA; inside ComfyUI the
        # model parks on the CPU between runs.
        cfg = dict(ckpt["model_config"])
        cfg["cradio_repo"] = paths["cradio_dir"]
        cfg["moge_checkpoint"] = paths["moge_checkpoint"]
        cfg["rvm_ckpt"] = paths["rvm_ckpt"]
        cfg["foundation_stats_path"] = paths["foundation_stats"]
        cfg["gradient_checkpoint"] = False
        cfg["device"] = "cpu"
        cfg["stage1_recurrent_64_sites"] = ()
        logger.info("loading CorridorKey v2 from %s", paths["weights"])
        with upstream.isolated_build_env():
            model = model_mod.CorridorKeyV3(**cfg)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        bad_missing = [k for k in missing if not k.startswith("foundations.")]
        bad_unexpected = [k for k in unexpected if not k.startswith("foundations.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "CorridorKey checkpoint did not match the model: "
                f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
        # Upstream silently disables RVM when its checkpoint path is wrong;
        # the model would still run, just worse. Refuse that instead.
        if model.foundations.rvm is None:
            raise RuntimeError(f"RVM failed to load from {paths['rvm_ckpt']}")
        model.eval()
        del ckpt
        _CACHE[key] = model
        return model


def unload_all():
    with _LOCK:
        _CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
