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
| **🦇 CorridorKey 2 Load Model** | Loads the model (1.27B parameters, about 5 GB). It waits in system RAM and only goes to the GPU while keying. |
| **🦇 CorridorKey 2** | The keyer. Inputs: `images`, an optional `hint` mask and optional `advanced` settings. Outputs: `foreground`, `alpha`, `preview`. |
| **🦇 CorridorKey 2 Advanced Params** | Optional. Window size, low-VRAM mode, hint carry, temp folder. If you don't connect it, the key node uses the same defaults. |

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
- **hint** (MASK, optional, white = subject):
  - one mask: a first-frame hint, which the model carries through the shot;
  - one mask per frame: a full hint;
  - no hint: the model picks the subject itself. That's fine for simple shots but less
    reliable when there are several possible subjects.

Outputs:

- `foreground`: straight (not premultiplied), despilled colour;
- `alpha`: the matte, 0 to 1;
- `preview`: the result over a checkerboard, for viewing.

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

Then restart ComfyUI. On first use, with `auto_download` on,
the loader fetches:

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
- **A hint at a different resolution is resized.** Upstream rejects a first-frame hint of
  the wrong size.
- **Download blocking stays local.** Upstream sets `HF_HUB_OFFLINE` and
  `TRANSFORMERS_OFFLINE` for the whole process; this pack sets them only while building
  the model, so other nodes can still download.
- **No clash with another MoGe.** The bundled MoGe is imported without replacing a `moge`
  another pack already loaded.
- **A missing RVM checkpoint is an error.** Upstream silently runs without RVM, which gives
  worse results.
