"""
The three nodes.

    🦇 CorridorKey 2 Load Model  ->  🦇 CorridorKey 2  <-  🦇 CorridorKey 2 Advanced Params (optional)

Node IDs and socket types carry a Bat_/BAT_ prefix: other ComfyUI CorridorKey 2
packs exist, and an unprefixed CORRIDORKEY2_MODEL socket would let their loader
be wired into this keyer.

The key node keeps only the controls an artist needs; everything about memory,
windows and hint carry lives on the Advanced Params node, whose defaults are
``driver.DEFAULTS``, so leaving it out of the graph gives the same result as
wiring it in untouched.

Widget order is part of the saved-workflow format: append new widgets at the
end, never in the middle.

Preview mode keys PREVIEW_FRAMES around one frame inside a normal queue run, so
with the queue's Run (On Change) mode every tweak re-keys just that window, and
upstream nodes stay cached.
"""

import logging
import os

import folder_paths
import numpy as np
from PIL import Image

from . import assets, driver

logger = logging.getLogger("Bat_CorridorKey2")

CATEGORY = "BAT/CorridorKey"
MODEL_TYPE = "BAT_CORRIDORKEY2_MODEL"
ADVANCED_TYPE = "BAT_CORRIDORKEY2_ADVANCED"

COLORSPACES = ["sRGB", "linear"]
SCREENS = ["green", "blue"]
HINT_MODES = ["all frames", "single frame"]
PREVIEW_FRAMES = 9   # window keyed around preview_frame: enough temporal context to look like the full key


def select_hint(hint, num_frames, hint_mode, hint_frame):
    """Resolve the MASK input against hint_mode -> (masks for the driver, hint frame).

    single frame takes a lone mask as the hint for ``hint_frame``, or picks that
    frame out of a per-frame batch (e.g. an automatic track).
    """
    if hint is None:
        return None, 0
    if hint.ndim == 2:
        hint = hint.unsqueeze(0)
    n = int(hint.shape[0])
    if hint_mode == "all frames":
        if n != num_frames:
            raise ValueError(
                f"hint_mode is 'all frames' but the hint has {n} mask(s) for {num_frames} "
                f"frames. Switch hint_mode to 'single frame' to use one mask.")
        return hint, 0
    k = min(int(hint_frame), num_frames - 1)
    if n == 1:
        return hint, k
    if n == num_frames:
        return hint[k:k + 1], k
    raise ValueError(
        f"the hint has {n} masks for {num_frames} frames. Give one mask, or one per frame.")


def preview_range(num_frames, preview_frame, hint_frame=None):
    """(frame shown, first, end) of the frames a preview keys.

    PREVIEW_FRAMES around the preview frame. A single hint (``hint_frame``) only
    reaches other frames through the carry, so the range stretches to include it.
    """
    p = min(int(preview_frame), num_frames - 1)
    end = min(num_frames, max(0, p - PREVIEW_FRAMES // 2) + PREVIEW_FRAMES)
    first = max(0, end - PREVIEW_FRAMES)
    if hint_frame is not None:
        first, end = min(first, hint_frame), max(end, hint_frame + 1)
    return p, first, end


def save_temp_previews(frames):
    """HxWx3 float frames in [0, 1] -> ComfyUI temp PNGs, as a node ``ui`` image list."""
    h, w = frames[0].shape[:2]
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        "bat_corridorkey2_preview", folder_paths.get_temp_directory(), w, h)
    out = []
    for i, f in enumerate(frames):
        filename = f"{name}_{counter + i:05}_.png"
        arr = (f.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(folder, filename), compress_level=1)
        out.append({"filename": filename, "subfolder": subfolder, "type": "temp"})
    return out


class CorridorKey2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Loads CorridorKey 2.0 (1.27B parameters, ~5 GB). On first use it fetches the pinned "
        "upstream runtime from GitHub and the weights from their publishers into "
        "models/corridorkey. The model waits in system RAM and only moves to the GPU while a "
        "CorridorKey 2 node is keying.")

    def load(self):
        return (assets.load_model(),)


class CorridorKey2AdvancedParams:
    @classmethod
    def INPUT_TYPES(cls):
        d = driver.DEFAULTS
        return {
            "required": {
                "window_frames": ("INT", {
                    "default": d["hann_chunk"], "min": 0, "max": 256,
                    "tooltip": "Time awareness: frames keyed together. 0 = the largest window that "
                               "fits the free VRAM (upstream's measured tables, up to 80). Bigger "
                               "is steadier over time and needs more VRAM; 1 keys every frame on "
                               "its own.",
                }),
                "low_vram": (["auto", "on", "off"], {
                    "default": d["low_vram"],
                    "tooltip": "Staged inference: intermediate features go to the system temp "
                               "folder so only one stage is on the GPU at a time. Slower; "
                               "practically the same result (alpha within ~0.01). auto = only "
                               "when the window would not fit otherwise.",
                }),
                "vram_limit_gb": ("FLOAT", {
                    "default": d["vram_limit_gb"], "min": 0.0, "max": 1024.0, "step": 0.5,
                    "tooltip": "Cap the VRAM the automatic window sizing plans for, in GB. "
                               "0 = everything free on the card. Use it on a shared GPU. Never "
                               "raises the budget above what is actually free.",
                }),
                "internal_resolution": (list(driver.SCALES), {
                    "default": d["internal_resolution"],
                    "tooltip": "Key a downscaled copy and scale the result back up to the plate "
                               "size (Lanczos). Faster and lighter, with softer edges and fine "
                               "detail (hair). The model was trained at native resolution.",
                }),
                "carry_hint": ("BOOLEAN", {
                    "default": d["carry_hint"],
                    "tooltip": "Feed the model's own matte forward as the hint for the frames each "
                               "window shares with the next. Turn off when a per-frame hint should "
                               "be taken literally on every frame.",
                }),
                "carry_hint_quality": ("FLOAT", {
                    "default": d["carry_hint_quality"], "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Internal hint strength: how much the model trusts its own carried "
                               "matte.",
                }),
            }
        }

    RETURN_TYPES = (ADVANCED_TYPE,)
    RETURN_NAMES = ("advanced",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Optional settings for CorridorKey 2. Unconnected, the key node uses these same "
        "defaults, so this is only needed to change time awareness, memory or hint carry.")

    def build(self, window_frames, **kw):
        return ({"hann_chunk": window_frames, **kw},)


class CorridorKey2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE, {"tooltip": "From 🦇 CorridorKey 2 Load Model."}),
                "images": ("IMAGE", {"tooltip": "The green/blue-screen clip, as a frame batch. "
                                                "Keyed at native resolution."}),
                "input_colorspace": (COLORSPACES, {
                    "default": "sRGB",
                    "tooltip": "sRGB = display-referred footage (MP4/ProRes/PNG). linear = "
                               "scene-linear Rec.709 plates (EXR). Convert ACEScg, log or PQ "
                               "upstream first. The foreground comes back in the same space: "
                               "clipped sRGB, or unclamped linear.",
                }),
                "screen": (SCREENS, {
                    "default": "green",
                    "tooltip": "Screen colour, for the despill only; the model keys either "
                               "without being told. Green is upstream's despill, blue is the "
                               "same formula mirrored.",
                }),
                "hint_quality": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to trust the hint mask: 0.95 an artist matte, 0.75 a "
                               "good automatic one (e.g. SeC), 0.5 or less a loose garbage matte. "
                               "Ignored without a hint.",
                }),
                "despill_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Spill removal on the foreground and preview. 0 = the model's "
                               "raw colour, 1 = full average despill. Changing it here re-keys "
                               "the whole clip; for instant changes set 0 and use 🦇 Despill "
                               "(ComfyUI-BAT-NodePack) after this node.",
                }),
                # Appended after the original widgets: widgets_values is positional.
                "hint_mode": (HINT_MODES, {
                    "default": "all frames",
                    "tooltip": "all frames = the hint is one mask per frame (a roto pass or an "
                               "automatic track). single frame = one hint on hint_frame, which "
                               "the model carries through the shot; from a per-frame batch it "
                               "takes the mask at hint_frame.",
                }),
                "hint_frame": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "The frame the hint belongs to (0 = first frame of the batch). "
                               "Above 0 the clip is keyed backwards and forwards from it, so a "
                               "matte on a clean middle frame guides the whole shot. Clamped to "
                               "the last frame.",
                }),
                "preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Key only a few frames around preview_frame and show the result on "
                               "the node, for fast tweaking (try the queue's Run (On Change) mode). "
                               "The outputs are then that single frame. Turn off for the full clip.",
                }),
                "preview_frame": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "The frame to preview. Keyed with its neighbours (9 frames) so it "
                               "looks like the full key. With a single-frame hint elsewhere in the "
                               "clip, everything between the two is keyed too.",
                }),
            },
            "optional": {
                "hint": ("MASK", {
                    "tooltip": "Optional subject hint, white = subject; see hint_mode. Without a "
                               "hint the model picks the subject itself.",
                }),
                "advanced": (ADVANCED_TYPE, {
                    "tooltip": "Optional 🦇 CorridorKey 2 Advanced Params. Unconnected = defaults."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "preview")
    OUTPUT_TOOLTIPS = (
        "Straight (unpremultiplied) foreground colour, despilled, in the input's colour space.",
        "The matte, 0 to 1.",
        "Despilled foreground over a checkerboard, sRGB, for viewing.",
    )
    FUNCTION = "key"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Green/blue-screen keying with CorridorKey 2.0 by Corridor Digital: alpha and "
        "foreground colour for a whole clip, temporally consistent. NVIDIA GPU only.")

    def key(self, model, images, input_colorspace, screen, hint_quality, despill_strength,
            hint_mode="all frames", hint_frame=0, preview=False, preview_frame=0,
            hint=None, advanced=None):
        import comfy.model_management as mm
        import comfy.utils

        hint, hint_frame = select_hint(hint, images.shape[0], hint_mode, hint_frame)
        if preview:
            single = hint is not None and hint.shape[0] == 1
            shown, first, end = preview_range(images.shape[0], preview_frame,
                                              hint_frame if single else None)
            images = images[first:end]
            if hint is not None and not single:
                hint = hint[first:end]
            hint_frame -= first if single else 0
        settings = dict(driver.DEFAULTS)
        if advanced:
            settings.update({k: v for k, v in advanced.items() if k in driver.DEFAULTS})

        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("CorridorKey 2 needs an NVIDIA GPU (RTX 30-series or newer).")
        # Free the whole card: the automatic window sizing reads what is free.
        mm.unload_all_models()
        mm.soft_empty_cache()

        state = {}

        def on_profile(chunk, stride, low_vram, estimate_gb, budget_gb, shape_hw, model_hw, frames):
            logger.info(
                "keying frames %d->%d of %dx%d at %dx%d (padded %dx%d): window %d/%d, "
                "low_vram=%s, estimated peak %.1f GB of %.1f GB budget",
                frames[0], frames[1], images.shape[2], images.shape[1], model_hw[1], model_hw[0],
                shape_hw[1], shape_hw[0], chunk, stride, low_vram, estimate_gb, budget_gb)
            if estimate_gb * 1.1 + 0.3 > budget_gb:
                logger.warning("even this window may not fit the GPU; expect out-of-memory")

        def on_total(n):
            state["pbar"] = comfy.utils.ProgressBar(n)

        def tick():
            state["pbar"].update(1)

        alpha, fg, comp = driver.key_clip(
            model, images, hint=hint, hint_frame=hint_frame,
            linear_input=(input_colorspace == "linear"),
            hint_quality=hint_quality, despill_strength=despill_strength, screen=screen,
            settings=settings, device=device, on_profile=on_profile, on_total=on_total,
            tick=tick, check_interrupt=mm.throw_exception_if_processing_interrupted)
        if not preview:
            return (fg, alpha, comp)
        i = shown - first
        fg, alpha, comp = fg[i:i + 1], alpha[i:i + 1], comp[i:i + 1]
        shown_images = save_temp_previews([comp[0], alpha[0, ..., None].expand(-1, -1, 3)])
        return {"ui": {"images": shown_images}, "result": (fg, alpha, comp)}


NODE_CLASS_MAPPINGS = {
    "Bat_CorridorKey2_LoadModel": CorridorKey2LoadModel,
    "Bat_CorridorKey2_Key": CorridorKey2,
    "Bat_CorridorKey2_AdvancedParams": CorridorKey2AdvancedParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Bat_CorridorKey2_LoadModel": "🦇 CorridorKey 2 Load Model",
    "Bat_CorridorKey2_Key": "🦇 CorridorKey 2",
    "Bat_CorridorKey2_AdvancedParams": "🦇 CorridorKey 2 Advanced Params",
}
