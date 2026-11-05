# ComfyUI-BAT-CorridorKey2

[CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0) green/blue-screen keying
inside ComfyUI. Give it a clip and get back the alpha and the foreground colour, keyed at
native resolution and stable across frames. A hint mask is optional.

CorridorKey 2.0 is by **Corridor Digital**. The model was designed and trained by Zachary
Harris, working with Niko Pueringer. This pack only connects it to ComfyUI. It is not
affiliated with or endorsed by Corridor Digital. If you key in Resolve, Nuke, After Effects
and so on, their official plugin is at [corridorkey.com](https://corridorkey.com).

## Nodes

| Node | What it does |
|---|---|
| **🦇 CorridorKey 2** | The keyer. Inputs: `images`, an optional `hint` mask and optional `advanced` settings. Outputs: `foreground`, `alpha`, `preview`. It loads the model itself (1.27B parameters, about 5 GB, fetched on first use); the model waits in system RAM between runs and only goes to the GPU while keying. |
| **🦇 CorridorKey 2 Advanced Params** | Optional. Time awareness (window), low-VRAM mode, a VRAM cap, internal resolution, hint carry. If you don't connect it, the key node uses the same defaults. |

### 🦇 CorridorKey 2

- **input_colorspace**: `sRGB` for display footage (MP4, ProRes, PNG); `linear` for
  scene-linear Rec.709 plates (EXR). Convert ACEScg, log or PQ material before this node.
  The foreground comes back in the same space: clipped sRGB, or unclamped linear with HDR
  values intact.
- **screen**: `green` or `blue`. This only affects the despill, because the model keys
  either colour without being told. Green uses upstream's despill; blue is the same
  formula mirrored.
- **hint_quality**: how far to trust the hint. 0.95 for an artist matte, 0.75 for a good
  automatic one (for example SeC), 0.5 or less for a loose garbage matte.
- **despill_strength**: 0 = the model's raw colour, 1 = full despill. Default 0.5.
- **hint** (MASK, optional, white = subject). With no hint the model picks the subject
  itself, which is fine for simple shots but less reliable when there are several
  possible subjects. The hint settings below only show while a mask is connected.
- **hint_mode**:
  - `all frames`: one mask per frame, such as a roto pass or an automatic track (SeC).
  - `single frame`: one hint on `hint_frame`, which the model carries through the shot.
    Given a per-frame batch, it takes the mask at `hint_frame`, so you can pick the best
    frame of an automatic track.
- **hint_frame** (only shown in `single frame`): 0 = the first frame. Above 0, the clip is
  keyed backwards and forwards from that frame, so a matte drawn on a clean frame in the
  middle of the shot guides all of it.

- **preview**: key only a few frames around **preview_frame** and show the result
  (composite and alpha) on the node. Combined with the queue's **Run (On Change)** mode,
  every tweak re-keys just that window in a few seconds. The preview frame is keyed with
  8 neighbours so it looks like the full key. A single-frame hint elsewhere in the clip
  pulls the frames in between into the preview too, because that's the only way the hint
  reaches it. While preview is on, the outputs are that one frame: turn it off for the
  full render.

Outputs:

- `foreground`: straight (not premultiplied), despilled colour;
- `alpha`: the matte, 0 to 1;
- `preview`: the result over a checkerboard, for viewing.

**Older workflows:** until 2026-11-05 the model came from a separate **🦇 CorridorKey 2 Load
Model** node. It is still installed, hidden from the node search, and the key node's `model`
socket is still there (optional, last) — so workflows that use it open and run unchanged. In
new graphs, leave `model` unconnected; you can delete the old loader and its link.

### 🦇 CorridorKey 2 Advanced Params

- **window_frames**: time awareness. 0 = the largest window that fits the free VRAM (up
  to 80 frames), 1 = every frame on its own.
- **low_vram**: `auto` uses it only when the window wouldn't fit otherwise.
- **vram_limit_gb**: plan for at most this much VRAM. 0 = everything free.
- **internal_resolution**: full, 3/4, 1/2 or 1/4.
- **carry_hint**, **carry_hint_quality**: the model's own matte carried between windows,
  and how far to trust it.

Low-VRAM mode spills its temporary files to the system temp folder; set `TMPDIR` to move it.

### Despill without re-keying, and why the foreground looks soft

Despill is the last thing the key node does to each frame, but as a widget there,
changing it re-keys the whole clip. For instant changes set **despill_strength** to 0
and put **🦇 Despill** from [ComfyUI-BAT-NodePack](https://github.com/BeeeFX/ComfyUI-BAT-NodePack)
after the key node: only that node re-runs. It outputs the despilled foreground, the
premultiplied foreground and a comp, and has an on-node viewer (comp / alpha / foreground /
premultiplied / spill removed / plate, zoom to real pixels, hold B to compare).
`tests/test_bat_despill_parity.py` checks it gives the same pixels as this pack's despill
and preview whenever both packs are installed.

The **foreground** output is straight (unpremultiplied) colour, meant to be multiplied by
alpha. Where alpha is near zero the colour has no effect in a comp, so the model carries the
subject's colour on past the matte edge (on a test shot, 10–20 px of hair colour where
alpha < 0.05), and nothing pins it from frame to frame: on its own it looks soft and
flickers at the edges. Multiplied by alpha — the preview, or 🦇 Despill's premultiplied and
comp outputs — the edge is crisp. Judge and render from those.

### Coming from the CorridorKey app

| App | Here |
|---|---|
| Input is Linear? | `input_colorspace` |
| Internal Resolution | `internal_resolution` (Advanced) |
| Use Time Awareness? / Time Awareness | `window_frames` (Advanced): 1 = off, 0 = auto |
| GPU Memory Limit | `vram_limit_gb` (Advanced) |
| Low GPU Memory | `low_vram` (Advanced) |
| Alpha Hint | the `hint` input |
| Hint Frames | `hint_mode` |
| Reference Frame | `hint_frame` |
| Alpha Hint Strength | `hint_quality` |
| Internal Hint Strength | `carry_hint_quality` (Advanced) |

## Requirements

- An NVIDIA GPU, RTX 30-series or newer. There is no CPU path.
- VRAM depends on the window (frames keyed together) and the resolution. With the window
  on `0` (auto) it picks the largest window that fits, using upstream's measured table:

  | GPU memory | 1080p | 1440p | 4K |
  |---|---|---|---|
  | 8 GB | 8/4 low-VRAM | 8/4 low-VRAM | 4/2 low-VRAM |
  | 12 GB | 24/12 low-VRAM | 8/4 | 8/4 low-VRAM |
  | 24 GB | 48/24 | 48/24 low-VRAM | 8/4 |
  | 48 GB | 80/40 | 80/40 | 48/24 low-VRAM |

  Bigger windows give better temporal consistency. Low-VRAM mode is slower and needs some
  disk space for temporary files. Its result is practically the same but not bit-identical:
  bf16 rounding takes a different path, which moved alpha by up to 0.008 in our tests.
- RAM: about 5 GB for the model, plus the clip itself (ComfyUI keeps frames as float32).
  A 100-frame 4K clip is about 10 GB in and 13 GB out.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/BeeeFX/ComfyUI-BAT-CorridorKey2
```

Then restart ComfyUI. On first use, the loader fetches:

1. **The CorridorKey 2.0 runtime**, pinned to an exact upstream commit, from GitHub into
   this pack's `upstream/` folder. Every file is checked against `ck2/UPSTREAM_MANIFEST.json`
   before anything is imported. This repository contains none of Corridor Digital's code.
2. **The weights, about 5 GB, from their original publishers** into `models/corridorkey/`:
   CorridorKey 2.0 (Hugging Face), C-RADIOv4-SO400M (NVIDIA), MoGe-2 ViT-B (Microsoft) and
   RVM small (Google DeepMind), each at the revision upstream pins.

Nothing in `requirements.txt` needs installing on a typical ComfyUI install; see the
comments in that file.

## Licence

**This pack is distributed under the Corridor Key Licence**, which is CC BY-NC-SA 4.0
plus additional terms. Section 5 of that licence requires it, so this pack is not MIT.
The full text is in `LICENSE`. In short, from upstream:

- You may use it for anything, including commercial work, and what you make with it is yours.
- You may not sell it, charge for access to it, or offer it as a paid API or inference
  service.
- Anything you build on it and release must be free, under the same licence, and keep the
  CorridorKey name.
- **Organisations with more than US $20M in annual revenue** (counted across the whole
  corporate group) **need a CorridorKey plugin seat for each person who uses it.** See
  [corridorkey.com](https://corridorkey.com).

The foundation models keep their own licences: NVIDIA Open Model License (C-RADIO), MIT
(MoGe), and Apache-2.0 for the code with CC-BY-4.0 for the weights (RVM). See `NOTICE`.

## How it relates to upstream

Upstream's `infer.py` reads clips from disk and writes video or EXR files. This pack runs
the same algorithm on ComfyUI tensors (`ck2/driver.py`), using upstream's own model, colour,
padding, windowing and despill functions. `tests/parity_vs_upstream.py` compares the two on
the same frames. At the pinned commit, the alpha is bit-identical to upstream's in sRGB,
linear-EXR and low-VRAM modes, and the foreground is within 2.4e-7 (float rounding).

Changes from upstream, all deliberate:

- **Blue-screen despill.**
- **A single hint on any frame** (`hint_frame`). Upstream only hints frame 0; above 0,
  this pack keys frames k→0 reversed and k→end, both starting from the hint.
- **Internal resolution.** Upstream's input path supports a model scale, but its output
  path refuses one. This pack scales the model output back up with Lanczos while it is
  still asinh-encoded, before the linear decode, so HDR highlights don't ring.
- **A VRAM cap** for the automatic window sizing.
- **A hint at a different resolution is resized.** Upstream rejects a first-frame hint of
  the wrong size.
- **Download blocking stays local.** Upstream sets `HF_HUB_OFFLINE` and
  `TRANSFORMERS_OFFLINE` for the whole process; this pack sets them only while building
  the model, so other nodes can still download.
- **No clash with another MoGe.** The bundled MoGe is imported without replacing a `moge`
  another pack already loaded.
- **A missing RVM checkpoint is an error.** Upstream silently runs without RVM, which gives
  worse results.
