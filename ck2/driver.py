"""
Tensor-in, tensor-out CorridorKey 2.0 inference.

Upstream's ``runtime.run_inference`` only reads clips from disk (OpenCV) and
only writes ffmpeg videos / EXR sequences. This module is the same algorithm
driven from ComfyUI tensors: it follows ``run_inference``,
``precompute_rvm_to_memmap``, ``precompute_foundations_to_memmap`` and
``hann_stage1_stream`` step for step, and calls upstream's own model, colour,
padding, windowing and despill functions rather than re-deriving them.
tests/parity_vs_upstream.py checks the result against upstream's infer.py.

What differs from upstream, on purpose:

* Frames come from an IMAGE batch, hints from a MASK batch (1 frame = first-frame
  hint, N frames = full hint). A hint at another resolution is resized the way
  upstream resizes a full hint video (Lanczos).
* Normal-mode RVM features stay in RAM instead of a temp memmap (they are small).
  Low-VRAM mode still spills the foundation features and the stage-1 baton to
  the system temp directory (``TMPDIR``) exactly as upstream does.
* The low-VRAM chunk sizes upstream writes onto the model are reset on every
  run, because here the model is cached between runs.
* Blue-screen despill, mirroring upstream's green formula (``screen="blue"``).
* ``internal_resolution``: upstream's preprocessing already supports a model scale
  (Lanczos before padding) but its render path refuses one. Here the model output
  is Lanczos-upscaled back to the plate size while still asinh-encoded, before the
  linear decode, so HDR highlights do not ring the way a linear-light resize would.
* ``vram_limit_gb`` caps the budget the automatic window sizing works from.
* A single hint can sit on any frame (``hint_frame``): the clip is keyed
  backwards and forwards from it (``plan_passes``). Upstream only hints frame 0.
"""

import math
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from . import upstream

SHAPE_QUANTUM = 128

# Defaults for everything the Advanced Params node can override. Single source
# of truth: the node reads these for its widget defaults.
DEFAULTS = {
    "hann_chunk": 0,                 # 0 = auto from free VRAM
    "low_vram": "auto",              # auto / on / off (not bit-exact vs off: bf16 path)
    "vram_limit_gb": 0.0,            # 0 = all free VRAM
    "internal_resolution": "full",   # see SCALES
    "carry_hint": True,
    "carry_hint_quality": 0.95,
}

# Upstream's own defaults for the low-VRAM chunking ("use the defaults unless you
# are actively tuning memory"), the stage-2 batch (1 measured no slower) and the
# window stride (half the window). Fixed, not exposed.
LOW_VRAM_CHUNKING = {
    "low_vram_enc_frame_chunk": 1,
    "low_vram_mlp_chunk_tokens": 65536,
    "low_vram_swin_window_batch": 16,
    "low_vram_full_attn_query_chunk_tokens": 2048,
}
LOW_VRAM_FOUNDATION_CHUNK = 4
STAGE2_BATCH = 1

SCALES = {"full": 1.0, "3/4": 0.75, "1/2": 0.5, "1/4": 0.25}


def _mods():
    return (upstream.submodule("runtime"), upstream.submodule("video_io"),
            upstream.submodule("color_utils"))


# ---------------------------------------------------------------------------
# Geometry and settings
# ---------------------------------------------------------------------------

def build_meta(num_frames, h, w, settings):
    """The subset of upstream's probe_*_meta dict the model path reads."""
    _, vio, _ = _mods()
    scale = SCALES[settings.get("internal_resolution", "full")]
    h_m, w_m = vio.scaled_hw(h, w, scale)
    h_out = vio.ceil_to_multiple(h_m, SHAPE_QUANTUM)
    w_out = vio.ceil_to_multiple(w_m, SHAPE_QUANTUM)
    pad_l = (w_out - w_m) // 2
    pad_t = (h_out - h_m) // 2
    return {
        "num_frames": int(num_frames),
        "native_hw": (h, w),
        "output_hw": (h, w),
        "model_native_hw": (h_m, w_m),
        "model_scale": scale,
        "shape_hw": (h_out, w_out),
        "pad": (pad_l, w_out - w_m - pad_l, pad_t, h_out - h_m - pad_t),
    }


def pick_profile(shape_hw, num_frames, budget_gb, chunk=0, low_vram="auto"):
    """(chunk, stride, low_vram, estimate_gb) for the requested settings.

    chunk=0 + low_vram=auto is upstream's own ``select_vram_profile``. The other
    combinations pin one of the two and let the measured tables pick the other.
    """
    rt, _, _ = _mods()
    if not chunk and low_vram == "auto":
        return rt.select_vram_profile(shape_hw, num_frames, budget_gb)
    mp = shape_hw[0] * shape_hw[1] / 1e6

    def fits(c, lv):
        est = rt.estimate_peak_vram_gb(mp, min(c, num_frames), lv)
        return est * rt._VRAM_MARGIN_SCALE + rt._VRAM_MARGIN_GB <= budget_gb, est

    if chunk:
        if low_vram == "auto":
            ok, est = fits(chunk, False)
            if ok:
                return chunk, max(1, chunk // 2), False, est
            return chunk, max(1, chunk // 2), True, fits(chunk, True)[1]
        lv = low_vram == "on"
        return chunk, max(1, chunk // 2), lv, fits(chunk, lv)[1]

    lv = low_vram == "on"
    sizes = rt._VRAM_PEAK_GB[lv][0]
    for c in reversed(sizes):
        ok, est = fits(c, lv)
        if ok:
            return c, max(1, c // 2), lv, est
    c = sizes[0]
    return c, max(1, c // 2), lv, fits(c, lv)[1]


def configure_chunking(model, low_vram):
    """Upstream's configure_low_vram_inference, plus the reset it never needs.

    The attributes it sets persist on the modules, and 0 means "off". A cached
    model that ran in low-VRAM mode once would otherwise keep chunking forever.
    """
    rt, _, _ = _mods()
    if low_vram:
        rt.configure_low_vram_inference(model, type("Cfg", (), LOW_VRAM_CHUNKING))
        return
    model.stage1.low_vram_enc_frame_chunk = 0
    for module in model.modules():
        name = module.__class__.__name__
        if name in ("SwiGLU", "FusionLinSwiglu"):
            module.inference_chunk_tokens = 0
        if name == "SwinAttention3D":
            module.inference_window_batch = 0
        if name == "CSwinAttention3D":
            module.inference_strip_batch = 0
        if name in ("FullSpatialAttention2D", "FullAttention3D"):
            module.inference_query_chunk_tokens = 0


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class FrameSource:
    """Per-frame model / foundation / review views, as upstream's readers make them.

    sRGB input follows the frame-directory path (display clipped to [0, 1],
    decoded to linear with the package gamma). Linear input follows the
    ``--frame_dir_linear`` path.
    """

    def __init__(self, images, meta, linear_input, frame_ids):
        self.images = images
        self.meta = meta
        self.linear = bool(linear_input)
        self.frame_ids = frame_ids      # pass-local t -> index into the batch
        self.cache = {}

    def _pad(self, arr):
        _, vio, _ = _mods()
        h_out, w_out = self.meta["shape_hw"]
        out, _ = vio.center_pad_to_shape(np.ascontiguousarray(arr), h_out, w_out)
        return out

    def views(self, t, need_model=True):
        _, vio, cu = _mods()
        frame = self.images[self.frame_ids[t], :, :, :3].float().cpu().numpy()
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        if self.linear:
            linear = frame
            display = np.clip(cu.linear_to_srgb(np.maximum(linear, 0.0)), 0.0, 1.0)
        else:
            display = np.clip(frame, 0.0, 1.0)
            linear = cu.srgb_to_linear(display)
        model_hw = tuple(self.meta["model_native_hw"])
        if model_hw != display.shape[:2]:
            # upstream preprocess_frame_dir_path: both views resized separately
            display = vio.resize_hwc_lanczos(display, model_hw)
            linear = vio.resize_hwc_lanczos(linear, model_hw)
        review = torch.from_numpy(
            self._pad(np.clip(display, 0.0, 1.0)).transpose(2, 0, 1)).to(torch.bfloat16)
        if not need_model:
            return None, review, review
        model = cu.linear_to_asinh(self._pad(linear))
        return torch.from_numpy(model.transpose(2, 0, 1)).to(torch.bfloat16), review, review

    def window(self, t0, t1):
        """Same contract and cache eviction as upstream's Sequential*WindowReader."""
        for t in range(t0, t1):
            if t not in self.cache:
                self.cache[t] = self.views(t)
        out = tuple(torch.stack([self.cache[t][i] for t in range(t0, t1)]) for i in range(3))
        for old in [k for k in self.cache if k < t0]:
            del self.cache[old]
        return out

    def chunk_foundation(self, t0, t1):
        return torch.stack([self.views(t, need_model=False)[1] for t in range(t0, t1)])


class HintSource:
    """MASK batch -> padded bf16 hint frames, as upstream's hint readers make them."""

    def __init__(self, masks, meta, frame_ids):
        self.masks = masks
        self.meta = meta
        self.frame_ids = frame_ids

    def frame(self, t):
        _, vio, _ = _mods()
        import cv2
        # A single mask is the hint for pass-local frame 0, whichever frame that is.
        idx = 0 if self.masks.shape[0] == 1 else self.frame_ids[t]
        hint = self.masks[idx].float().cpu().numpy().astype(np.float32, copy=False)
        h_model, w_model = self.meta["model_native_hw"]
        if hint.shape[:2] != (h_model, w_model):
            hint = cv2.resize(hint, (w_model, h_model), interpolation=cv2.INTER_LANCZOS4)
        h_out, w_out = self.meta["shape_hw"]
        hint, _ = vio.center_pad_to_shape(
            np.ascontiguousarray(np.clip(hint, 0.0, 1.0)), h_out, w_out)
        return torch.from_numpy(hint[None]).to(dtype=torch.bfloat16)

    def window(self, t0, t1):
        return torch.stack([self.frame(t) for t in range(t0, t1)])


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def despill(fg, strength, screen):
    """Upstream's average despill for green; the same formula mirrored for blue."""
    rt, _, _ = _mods()
    if screen == "green":
        return rt._despill_average_srgb_torch(fg, strength=strength)
    if strength <= 0.0:
        return fg
    r, g, b = fg[:, 0:1], fg[:, 1:2], fg[:, 2:3]
    spill = (b - (r + g) * 0.5).clamp(min=0.0)
    full = torch.cat([r + spill * 0.5, g + spill * 0.5, b - spill], dim=1)
    if strength >= 1.0:
        return full
    return fg * (1.0 - strength) + full * strength


def _upscale_to_output(x, meta):
    """Model-resolution [B,C,h,w] -> plate size, Lanczos (cv2), on the raw model output.

    Runs on alpha + asinh-encoded foreground, so the resize happens in the model's
    own compressed HDR domain rather than in linear light.
    """
    if tuple(meta["model_native_hw"]) == tuple(meta["output_hw"]):
        return x
    _, vio, _ = _mods()
    arr = x.detach().float().cpu().numpy()
    up = np.stack([vio.resize_chw_lanczos(np.ascontiguousarray(f), tuple(meta["output_hw"]))
                   for f in arr])
    return torch.from_numpy(up).to(x.device)


def render(logits, meta, *, linear_output, despill_strength, screen, checker_lin):
    """Stage-2 logits -> (alpha, foreground, preview), all float32 on logits' device.

    linear_output=True follows upstream's EXR path: unclamped scene-linear
    foreground, despilled in linear light (their ``cutout`` EXR, RGB part).
    linear_output=False follows the video path: sRGB foreground clipped to
    [0, 1], despilled in sRGB (their ``cutout`` video, RGB part). The preview is
    upstream's checker composite, always returned sRGB-encoded for viewing.
    """
    rt, _, cu = _mods()
    out = _upscale_to_output(rt._crop_native_torch(logits[:, 0:4].float(), meta), meta)
    alpha = out[:, 0:1].clamp(0.0, 1.0)
    fg_lin = cu.asinh_to_linear(out[:, 1:4])
    if linear_output:
        fg = despill(fg_lin, despill_strength, screen)
        comp_lin = fg * alpha + checker_lin * (1.0 - alpha)
    else:
        pf = cu.linear_to_srgb(fg_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        fg = despill(pf, despill_strength, screen)
        comp_lin = cu.srgb_to_linear(fg) * alpha + checker_lin * (1.0 - alpha)
    preview = cu.linear_to_srgb(comp_lin.clamp(min=0.0)).clamp(0.0, 1.0)
    return alpha[:, 0], fg.permute(0, 2, 3, 1), preview.permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _noop(*_a, **_k):
    pass


@torch.no_grad()
def _rvm_prepass(model, src, meta, chunk, tick):
    """precompute_rvm_to_memmap, kept in RAM: RVM runs once, in order, over the clip."""
    device = next(model.foundations.rvm.parameters()).device
    T = meta["num_frames"]
    H, W = meta["shape_hw"]
    H_e, W_e = H // 4, W // 4
    out = torch.empty((T, model.foundations.rvm.embed_dim, H_e // 16, W_e // 16),
                      dtype=torch.float16)
    mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
    std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
    state = None
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        Tc = t1 - t0
        rgb = src.chunk_foundation(t0, t1).unsqueeze(0).to(device, non_blocking=True).float()
        rgb_rvm = F.interpolate(
            rgb.reshape(Tc, 3, H, W), size=(H_e, W_e),
            mode="bilinear", align_corners=False,
        ).reshape(1, Tc, 3, H_e, W_e).clamp(0.0, 1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats, state = model.foundations.rvm(rgb_rvm, state=state)
        out[t0:t1] = ((feats.float().cpu() - mean) / std)[0].to(torch.float16)
        del rgb, rgb_rvm, feats
        tick()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def _foundation_prepass(model, src, meta, prefix, chunk, tick):
    """precompute_foundations_to_memmap (low-VRAM mode only)."""
    rt, _, _ = _mods()
    device = next(model.foundations.parameters()).device
    T = meta["num_frames"]
    H, W = meta["shape_hw"]
    h64, w64 = H // 64, W // 64
    C_cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
    C_moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
    if C_moge <= 0:
        raise RuntimeError("MoGe is required for this model")
    paths = {"moge": prefix + "_moge.f32", "moge_shape": (T, C_moge, h64, w64)}
    moge_out = np.memmap(paths["moge"], dtype=np.float32, mode="w+", shape=paths["moge_shape"])
    cradio_out = None
    if C_cradio > 0:
        paths["cradio"] = prefix + "_cradio.f32"
        paths["cradio_shape"] = (T, C_cradio, h64, w64)
        cradio_out = np.memmap(paths["cradio"], dtype=np.float32, mode="w+",
                               shape=paths["cradio_shape"])
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        Tc = t1 - t0
        rgb = src.chunk_foundation(t0, t1).unsqueeze(0).to(device, non_blocking=True)
        rgb_radio, rgb_moge = rt._build_foundation_inputs(rgb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ff = model.foundations(
                rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
            )
        moge_out[t0:t1] = ff["moge"].float().cpu().numpy()
        if cradio_out is not None:
            cradio_out[t0:t1] = ff["cradio"].float().cpu().numpy()
        del rgb, rgb_radio, rgb_moge, ff
        tick()
    moge_out.flush()
    del moge_out
    if cradio_out is not None:
        cradio_out.flush()
        del cradio_out
    torch.cuda.empty_cache()
    return paths


@torch.no_grad()
def _stage1_stream(model, src, hints, hint_mode, meta, *, chunk, stride, hint_quality,
                   carry_hint, carry_hint_quality, rvm_feats, foundation_paths,
                   baton_path, on_baton_frames, tick):
    """hann_stage1_stream with tensor sources. Comments mark the upstream lines."""
    _, vio, _ = _mods()
    device = next(model.stage1.parameters()).device
    T = meta["num_frames"]
    H, W = meta["shape_hw"]
    d_16 = model.stage1.d_16
    starts = vio.window_starts(T, chunk, stride)
    overlap = max(0, chunk - stride)
    baton_fp16 = None
    if on_baton_frames is None:
        baton_fp16 = np.memmap(baton_path, dtype=np.float16, mode="w+",
                               shape=(T, d_16, H // 16, W // 16))
    live_acc, live_w = {}, {}
    cradio_map = moge_map = None
    if foundation_paths is not None:
        moge_map = np.memmap(foundation_paths["moge"], dtype=np.float32, mode="r",
                             shape=tuple(foundation_paths["moge_shape"]))
        if model.stage1.cradio_dim > 0:
            cradio_map = np.memmap(foundation_paths["cradio"], dtype=np.float32, mode="r",
                                   shape=tuple(foundation_paths["cradio_shape"]))
    initial_hint = hints.frame(0) if hint_mode == "first" else None
    carry_cache = {}
    stage2_frame_cache = {}

    def flush_before(t_limit):
        ready = []
        for k in sorted(k for k in live_acc if k < t_limit):
            frame = (live_acc.pop(k) / max(live_w.pop(k), 1e-6)).astype(np.float16)
            if on_baton_frames is None:
                baton_fp16[k] = frame
            else:
                ready.append((k, frame, stage2_frame_cache.pop(k)))
        if ready:
            on_baton_frames(ready)

    for i, t0 in enumerate(starts):
        t1 = min(t0 + chunk, T)
        Tc = t1 - t0
        model_chunk, display_chunk, _review = src.window(t0, t1)
        hint_cpu = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16)
        hint_quality_cpu = torch.zeros(1, Tc, dtype=torch.bfloat16)
        hint_present = torch.zeros(1, Tc, dtype=torch.bool)
        if hint_mode == "full":
            hint_cpu[0] = hints.window(t0, t1)
            hint_quality_cpu.fill_(float(hint_quality))
            hint_present.fill_(True)
        for j, t_abs in enumerate(range(t0, t1)):
            if t_abs in carry_cache:
                hint_cpu[0, j] = carry_cache[t_abs]
                hint_quality_cpu[0, j] = float(carry_hint_quality)
                hint_present[0, j] = True
        if initial_hint is not None and t0 == 0:
            hint_cpu[0, 0] = initial_hint.to(dtype=hint_cpu.dtype)
            hint_quality_cpu[0, 0] = float(hint_quality)
            hint_present[0, 0] = True
        if not bool(hint_present.any().item()):
            hint_quality_cpu.fill_(-1.0)   # "no hint in this window: pick the subject"

        if on_baton_frames is not None:
            for j, t_abs in enumerate(range(t0, t1)):
                stage2_frame_cache[t_abs] = model_chunk[j]

        rgb_c = model_chunk.unsqueeze(0).to(device, non_blocking=True)
        hint_c = hint_cpu.to(device, non_blocking=True)
        hint_quality_c = hint_quality_cpu.to(device, non_blocking=True)
        if foundation_paths is None:
            rt, _, _ = _mods()
            rgb_found = display_chunk.unsqueeze(0).to(device, non_blocking=True)
            rgb_radio, rgb_moge = rt._build_foundation_inputs(rgb_found)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ff = model.foundations(
                    rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                    rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                )
            cradio_feat = ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64)
            moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            del ff, rgb_found, rgb_radio, rgb_moge
        else:
            moge_feat = torch.from_numpy(np.asarray(moge_map[t0:t1]).copy())
            moge_feat = moge_feat.unsqueeze(0).to(device, non_blocking=True).float()
            cradio_feat = torch.from_numpy(np.asarray(cradio_map[t0:t1]).copy())
            cradio_feat = cradio_feat.unsqueeze(0).to(device, non_blocking=True).float()
        rvm_feat = rvm_feats[t0:t1].unsqueeze(0).to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s1 = model.stage1(
                rgb_c, hint_c, moge_feat, cradio_feat=cradio_feat,
                rvm_feat=rvm_feat, t_stride=1.0, hint_quality=hint_quality_c,
            )
        baton_gpu = s1["baton"][0].float()
        if carry_hint:
            aux_full = F.interpolate(
                s1["aux"][0, :, 0:1].float(), size=(H, W), mode="bilinear",
                align_corners=False).clamp(0.0, 1.0)
            aux_cpu = aux_full.to("cpu", dtype=torch.bfloat16)
            for j, t_abs in enumerate(range(t0, t1)):
                carry_cache[t_abs] = aux_cpu[j].clone()
            next_start = starts[i + 1] if i + 1 < len(starts) else T
            for old_t in [k for k in carry_cache if k < next_start]:
                del carry_cache[old_t]
            del aux_full, aux_cpu
        w = vio.taper_weights(Tc, overlap, is_first=(i == 0), is_last=(i == len(starts) - 1),
                              device="cpu", dtype=torch.float32)
        weighted = (baton_gpu * w.to(device).view(Tc, 1, 1, 1)).to("cpu").numpy()
        w_np = w.numpy()
        for j, t_abs in enumerate(range(t0, t1)):
            if t_abs not in live_acc:
                live_acc[t_abs] = weighted[j].copy()
                live_w[t_abs] = float(w_np[j])
            else:
                live_acc[t_abs] += weighted[j]
                live_w[t_abs] += float(w_np[j])
        next_start = starts[i + 1] if i + 1 < len(starts) else T
        del s1, baton_gpu, weighted, cradio_feat, moge_feat, rvm_feat
        del rgb_c, hint_c, hint_quality_c, hint_cpu, hint_quality_cpu
        del model_chunk, display_chunk, _review
        flush_before(next_start)
        tick()
    torch.cuda.synchronize()
    flush_before(T + 1)
    if baton_fp16 is not None:
        baton_fp16.flush()
    del baton_fp16, live_acc, live_w, cradio_map, moge_map
    torch.cuda.empty_cache()
    return len(starts)


def count_steps(num_frames, chunk, stride, low_vram):
    _, vio, _ = _mods()
    n = math.ceil(num_frames / max(1, min(chunk, 8)))                      # RVM prepass
    n += len(vio.window_starts(num_frames, chunk, stride))                 # stage 1
    if low_vram:
        n += math.ceil(num_frames / LOW_VRAM_FOUNDATION_CHUNK)
        n += math.ceil(num_frames / STAGE2_BATCH)
    return n


def _run_pass(model, src, hints, hint_mode, meta, profile, s, outs, *, linear_input,
              hint_quality, despill_strength, screen, tmp, step):
    """One pass of run_inference over ``src.frame_ids``, writing into ``outs``."""
    rt, vio, cu = _mods()
    chunk, stride, low_vram = profile
    T = meta["num_frames"]
    H, W = meta["output_hw"]
    ids = src.frame_ids
    alpha_out, fg_out, prev_out = outs
    baton_path = os.path.join(tmp, "baton.f16")
    stage2_batch = STAGE2_BATCH
    checker_lin = None

    def write_prediction_batches(items):
        nonlocal checker_lin
        dev = next(model.stage2.parameters()).device
        if checker_lin is None:
            checker = torch.from_numpy(vio.make_checker(H, W)).to(dev).unsqueeze(0)
            checker_lin = cu.srgb_to_linear(checker)
        for offset in range(0, len(items), stage2_batch):
            sub = items[offset:offset + stage2_batch]
            rgb_t = torch.stack([it[2] for it in sub], dim=0).unsqueeze(0).to(dev, non_blocking=True)
            baton_t = torch.from_numpy(np.stack([it[1] for it in sub], axis=0))
            baton_t = baton_t.unsqueeze(0).to(dev, non_blocking=True).float()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.stage2(rgb_t, baton_t)
                if isinstance(logits, tuple):
                    logits = logits[0]
            a, fg, pv = render(logits[0], meta, linear_output=linear_input,
                               despill_strength=float(despill_strength), screen=screen,
                               checker_lin=checker_lin)
            idx = [ids[it[0]] for it in sub]
            alpha_out[idx] = a.cpu()
            fg_out[idx] = fg.cpu()
            prev_out[idx] = pv.cpu()
            del logits, baton_t, rgb_t, a, fg, pv
            if low_vram:
                step()

    rt._phase1_setup(model, stage1_on_gpu=not low_vram)
    configure_chunking(model, low_vram)
    rvm_feats = _rvm_prepass(model, src, meta, max(1, min(chunk, 8)), step)
    foundation_paths = None
    if low_vram:
        foundation_paths = _foundation_prepass(
            model, src, meta, os.path.join(tmp, "foundation"),
            LOW_VRAM_FOUNDATION_CHUNK, step)
        model.foundations.cpu()
        model.stage2.cpu()
        model.stage1.cuda()
        torch.cuda.empty_cache()
    else:
        model.stage2.cuda()
        model.stage2.eval()
        torch.cuda.empty_cache()

    _stage1_stream(
        model, src, hints, hint_mode, meta, chunk=chunk, stride=stride,
        hint_quality=hint_quality, carry_hint=bool(s["carry_hint"]),
        carry_hint_quality=float(s["carry_hint_quality"]), rvm_feats=rvm_feats,
        foundation_paths=foundation_paths, baton_path=baton_path,
        on_baton_frames=None if low_vram else write_prediction_batches, tick=step)
    del rvm_feats

    if low_vram:
        rt._phase2_setup(model)
        baton = np.memmap(baton_path, dtype=np.float16, mode="r",
                          shape=(T, model.stage1.d_16, meta["shape_hw"][0] // 16,
                                 meta["shape_hw"][1] // 16))
        try:
            for t0 in range(0, T, stage2_batch):
                t1 = min(T, t0 + stage2_batch)
                model_chunk, _f, _r = src.window(t0, t1)
                write_prediction_batches([
                    (t, np.asarray(baton[t]).copy(), model_chunk[t - t0])
                    for t in range(t0, t1)])
                del model_chunk, _f, _r
        finally:
            del baton
    src.cache.clear()


def plan_passes(num_frames, hint_mode, hint_frame):
    """Frame orders to key. A single hint on frame k > 0 keys k..0 (reversed), then k..end.

    Both passes start from the hint, so each half carries it outwards from frame k;
    the forward pass runs last and owns frame k itself.
    """
    k = min(max(0, int(hint_frame)), num_frames - 1)
    if hint_mode != "first" or k == 0:
        return [list(range(num_frames))]
    return [list(range(k, -1, -1)), list(range(k, num_frames))]


def key_clip(model, images, *, hint=None, hint_frame=0, linear_input=False, hint_quality=0.95,
             despill_strength=0.5, screen="green", settings=None, device=None,
             on_profile=_noop, on_total=_noop, tick=_noop, check_interrupt=_noop):
    """Key an IMAGE batch. Returns (alpha [T,H,W], foreground [T,H,W,3], preview [T,H,W,3]).

    ``hint`` is an optional MASK batch: 1 frame = a single hint, placed on
    ``hint_frame`` (0-based); T frames = one hint per frame (``hint_frame`` unused).
    ``device`` is the CUDA device to run on (default: current).
    """
    s = dict(DEFAULTS)
    s.update(settings or {})
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"expected an IMAGE batch [T,H,W,C>=3], got {tuple(images.shape)}")
    T, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])

    hint_mode = None
    if hint is not None:
        if hint.ndim == 2:
            hint = hint.unsqueeze(0)
        if hint.shape[0] == 1:
            hint_mode = "first"
        elif hint.shape[0] == T:
            hint_mode = "full"
        else:
            raise ValueError(
                f"hint has {hint.shape[0]} frames but the clip has {T}. Give one mask "
                f"(a single hint) or exactly one mask per frame.")

    device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        free_b, _ = torch.cuda.mem_get_info()
        budget_gb = (free_b + torch.cuda.memory_reserved()) / 2 ** 30
        if float(s["vram_limit_gb"] or 0) > 0:
            budget_gb = min(budget_gb, float(s["vram_limit_gb"]))

        passes = []
        for ids in plan_passes(T, hint_mode, hint_frame):
            meta = build_meta(len(ids), H, W, s)
            chunk, stride, low_vram, estimate = pick_profile(
                meta["shape_hw"], len(ids), budget_gb, int(s["hann_chunk"]), s["low_vram"])
            on_profile(chunk=chunk, stride=stride, low_vram=low_vram,
                       estimate_gb=estimate, budget_gb=budget_gb, shape_hw=meta["shape_hw"],
                       model_hw=meta["model_native_hw"], frames=(ids[0], ids[-1]))
            passes.append((ids, meta, (chunk, stride, low_vram)))
        on_total(sum(count_steps(len(ids), c, st, lv) for ids, _, (c, st, lv) in passes))

        def step():
            check_interrupt()
            tick()

        outs = (torch.empty((T, H, W), dtype=torch.float32),
                torch.empty((T, H, W, 3), dtype=torch.float32),
                torch.empty((T, H, W, 3), dtype=torch.float32))
        tmp = tempfile.mkdtemp(prefix="bat_corridorkey2_")   # low-VRAM spill; honours TMPDIR
        try:
            for ids, meta, profile in passes:
                src = FrameSource(images, meta, linear_input, ids)
                hints = HintSource(hint, meta, ids) if hint is not None else None
                _run_pass(model, src, hints, hint_mode, meta, profile, s, outs,
                          linear_input=linear_input, hint_quality=hint_quality,
                          despill_strength=despill_strength, screen=screen, tmp=tmp, step=step)
        finally:
            model.foundations.cpu()
            model.stage1.cpu()
            model.stage2.cpu()
            shutil.rmtree(tmp, ignore_errors=True)
            torch.cuda.empty_cache()
    return outs
