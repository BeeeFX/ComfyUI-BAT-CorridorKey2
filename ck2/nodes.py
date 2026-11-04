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
"""

import logging

from . import assets, driver

logger = logging.getLogger("Bat_CorridorKey2")

CATEGORY = "BAT/CorridorKey"
MODEL_TYPE = "BAT_CORRIDORKEY2_MODEL"
ADVANCED_TYPE = "BAT_CORRIDORKEY2_ADVANCED"

COLORSPACES = ["sRGB", "linear"]
SCREENS = ["green", "blue"]


class CorridorKey2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fetch what is missing on first use: the pinned CorridorKey 2.0 "
                               "runtime from GitHub (into this pack's upstream/ folder) and ~5 GB "
                               "of weights from their original publishers into models/corridorkey. "
                               "Turn off on air-gapped machines; the node then errors with what "
                               "is missing and where it goes.",
                }),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Loads CorridorKey 2.0 (1.27B parameters, ~5 GB). The model waits in system RAM "
        "and only moves to the GPU while a CorridorKey 2 node is keying.")

    def load(self, auto_download):
        return (assets.load_model(allow_download=auto_download),)


class CorridorKey2AdvancedParams:
    @classmethod
    def INPUT_TYPES(cls):
        d = driver.DEFAULTS
        return {
            "required": {
                "window_frames": ("INT", {
                    "default": d["hann_chunk"], "min": 0, "max": 256,
                    "tooltip": "Frames keyed together in stage 1. 0 = pick the largest window "
                               "that fits the free VRAM (upstream's measured tables). Bigger is "
                               "more temporally stable and needs more VRAM; 1 keys every frame "
                               "on its own.",
                }),
                "window_stride": ("INT", {
                    "default": d["hann_stride"], "min": 0, "max": 256,
                    "tooltip": "Frames between window starts. 0 = half the window (50% overlap, "
                               "upstream's default). Capped at window_frames.",
                }),
                "low_vram": (["auto", "on", "off"], {
                    "default": d["low_vram"],
                    "tooltip": "Staged inference: foundation features and the stage-1 result "
                               "go to temp_dir on disk so only one stage is on the GPU at a "
                               "time. Slower; practically the same result (bf16 rounding differs, "
                               "alpha within ~0.01). auto = use it only when the window would "
                               "not fit otherwise.",
                }),
                "carry_hint": ("BOOLEAN", {
                    "default": d["carry_hint"],
                    "tooltip": "Feed the model's own matte from each window forward as the hint "
                               "for the overlapping frames of the next one. Turn off when a "
                               "per-frame hint should be taken literally on every frame.",
                }),
                "carry_hint_quality": ("FLOAT", {
                    "default": d["carry_hint_quality"], "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much the model trusts its own carried matte.",
                }),
                "stage2_batch": ("INT", {
                    "default": d["stage2_batch"], "min": 1, "max": 32,
                    "tooltip": "Full-resolution frames refined per stage-2 call. Upstream found "
                               "1 no slower than larger batches, and it uses the least VRAM.",
                }),
                "temp_dir": ("STRING", {
                    "default": d["temp_dir"],
                    "tooltip": "Where low-VRAM mode spills features (can be several GB on long "
                               "4K clips). Empty = the system temp directory. Cleaned up after "
                               "every run.",
                }),
                "free_comfy_vram": ("BOOLEAN", {
                    "default": d["free_comfy_vram"],
                    "tooltip": "Unload ComfyUI's other models before keying, so the automatic "
                               "window sizing sees the whole card.",
                }),
                "keep_model_on_gpu": ("BOOLEAN", {
                    "default": d["keep_model_on_gpu"],
                    "tooltip": "Leave CorridorKey on the GPU after the run (faster re-runs, "
                               "~5 GB of VRAM held). Off = park it back in system RAM.",
                }),
                "low_vram_foundation_chunk": ("INT", {
                    "default": d["low_vram_foundation_chunk"], "min": 1, "max": 64,
                    "tooltip": "Low-VRAM: frames per foundation-model pass."}),
                "low_vram_enc_frame_chunk": ("INT", {
                    "default": d["low_vram_enc_frame_chunk"], "min": 0, "max": 64,
                    "tooltip": "Low-VRAM: stage-1 encoder frame chunking (0 = off)."}),
                "low_vram_mlp_chunk_tokens": ("INT", {
                    "default": d["low_vram_mlp_chunk_tokens"], "min": 0, "max": 1 << 22,
                    "tooltip": "Low-VRAM: MLP token chunking (0 = off)."}),
                "low_vram_swin_window_batch": ("INT", {
                    "default": d["low_vram_swin_window_batch"], "min": 0, "max": 4096,
                    "tooltip": "Low-VRAM: Swin window / strip batching (0 = off)."}),
                "low_vram_full_attn_query_chunk_tokens": ("INT", {
                    "default": d["low_vram_full_attn_query_chunk_tokens"], "min": 0, "max": 1 << 20,
                    "tooltip": "Low-VRAM: full-attention query chunking (0 = off)."}),
                "linear_brightness": ("FLOAT", {
                    "default": d["linear_brightness"], "min": 0.01, "max": 100.0, "step": 0.01,
                    "tooltip": "Experimental (upstream): gain applied in linear light before the "
                               "model, undone on the foreground. Leave at 1 for production.",
                }),
                "linear_contrast": ("FLOAT", {
                    "default": d["linear_contrast"], "min": 0.01, "max": 100.0, "step": 0.01,
                    "tooltip": "Experimental (upstream): contrast around the pivot, undone on "
                               "the foreground. Leave at 1 for production.",
                }),
                "linear_contrast_pivot": ("FLOAT", {
                    "default": d["linear_contrast_pivot"], "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Experimental (upstream): pivot for linear_contrast."}),
            }
        }

    RETURN_TYPES = (ADVANCED_TYPE,)
    RETURN_NAMES = ("advanced",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Optional settings for CorridorKey 2. Unconnected, the key node uses these same "
        "defaults, so this is only needed to change memory, window or hint-carry behaviour.")

    def build(self, window_frames, window_stride, **kw):
        return ({"hann_chunk": window_frames, "hann_stride": window_stride, **kw},)


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
                               "raw colour, 1 = full average despill.",
                }),
            },
            "optional": {
                "hint": ("MASK", {
                    "tooltip": "Optional subject hint, white = subject. One mask = first-frame "
                               "hint (the model carries it through the shot). One mask per "
                               "frame = full hint. Without a hint the model picks the subject.",
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
            hint=None, advanced=None):
        import comfy.model_management as mm
        import comfy.utils

        settings = dict(driver.DEFAULTS)
        if advanced:
            settings.update({k: v for k, v in advanced.items() if k in driver.DEFAULTS})

        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("CorridorKey 2 needs an NVIDIA GPU (RTX 30-series or newer).")
        if settings["free_comfy_vram"]:
            mm.unload_all_models()
            mm.soft_empty_cache()

        state = {}

        def on_profile(chunk, stride, low_vram, estimate_gb, budget_gb, shape_hw):
            logger.info(
                "keying %d frames at %dx%d (padded %dx%d): window %d/%d, low_vram=%s, "
                "estimated peak %.1f GB of %.1f GB available",
                images.shape[0], images.shape[2], images.shape[1], shape_hw[1], shape_hw[0],
                chunk, stride, low_vram, estimate_gb, budget_gb)
            if estimate_gb * 1.1 + 0.3 > budget_gb:
                logger.warning("even this window may not fit the GPU; expect out-of-memory")

        def on_total(n):
            state["pbar"] = comfy.utils.ProgressBar(n)

        def tick():
            state["pbar"].update(1)

        alpha, fg, preview = driver.key_clip(
            model, images, hint=hint, linear_input=(input_colorspace == "linear"),
            hint_quality=hint_quality, despill_strength=despill_strength, screen=screen,
            settings=settings, device=device, on_profile=on_profile, on_total=on_total,
            tick=tick, check_interrupt=mm.throw_exception_if_processing_interrupted)
        return (fg, alpha, preview)


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
