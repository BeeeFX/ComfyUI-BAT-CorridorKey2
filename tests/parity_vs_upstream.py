"""Parity: this pack's tensor driver vs upstream's own infer.py, on the same frames.

    python tests/parity_vs_upstream.py --upstream_repo <checkout with weights/ + vfms/> \
        --work <scratch dir> [--ffmpeg_bin <ffmpeg>] [--keep]

Needs a CUDA GPU and the weights. Builds a small synthetic green-screen clip
(500x300, so the 128-quantum padding is exercised on both axes; 20 frames with an
8/4 window, so window overlap, the Hann taper and hint carry are all exercised),
runs upstream in EXR/float mode, runs ck2.driver.key_clip on the identical
decoded frames, and compares:

  A  sRGB PNG input + first-frame hint, normal mode
  B  linear EXR input (--frame_dir_linear) + full per-frame hint, normal mode
  C  case A in low-VRAM mode
  D  case A run twice through upstream: the GPU's own run-to-run noise floor

Upstream writes raw outputs; the pack's foreground/preview are derived from
those with upstream's own functions, so every comparison is like for like.
Nothing here imports or starts ComfyUI beyond its folder_paths module.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PACK))
sys.path.insert(0, PACK)
sys.path.insert(1, COMFY)

T, H, W = 20, 300, 500


def make_clip(work):
    """A drifting, textured, soft-edged blob with a thin hair-like stroke over noisy green."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames, mattes = [], []
    for t in range(T):
        bg = np.empty((H, W, 3), np.float32)
        bg[..., 0], bg[..., 1], bg[..., 2] = 0.12, 0.62, 0.22
        bg += rng.normal(0, 0.02, (H, W, 1)).astype(np.float32) * np.array([1, 1.4, 1], np.float32)
        bg *= (0.85 + 0.15 * (xx / W))[..., None]                       # uneven lighting
        cx, cy = 170 + 7 * t, 150 + 25 * np.sin(t / 4)
        d = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.3) ** 2)
        a = np.clip((70 - d) / 6, 0, 1)                                    # soft edge
        stroke = np.exp(-((xx - (cx + 60 + 0.4 * (yy - cy))) ** 2) / 2.0) * (np.abs(yy - cy) < 60)
        a = np.maximum(a, 0.8 * stroke.astype(np.float32))
        tex = 0.5 + 0.5 * np.sin(xx / 9 + t) * np.cos(yy / 7)
        fg = np.stack([0.85 * tex + 0.1, 0.55 * tex + 0.1, 0.45 * tex + 0.08], -1)
        spill = 0.15 * (1 - a)[..., None] * np.array([0, 1, 0], np.float32)  # green fringe
        img = np.clip(fg * a[..., None] + bg * (1 - a[..., None]) + spill * a[..., None], 0, 1)
        frames.append(img.astype(np.float32))
        mattes.append(a.astype(np.float32))
    d_png, d_exr, d_hint = (os.path.join(work, n) for n in ("png", "exr", "hint_full"))
    for d in (d_png, d_exr, d_hint):
        os.makedirs(d, exist_ok=True)
    for t, (img, m) in enumerate(zip(frames, mattes)):
        u8 = (img * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(d_png, f"f_{t:04d}.png"), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
        lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)
        cv2.imwrite(os.path.join(d_exr, f"f_{t:04d}.exr"), cv2.cvtColor(lin, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        cv2.imwrite(os.path.join(d_hint, f"h_{t:04d}.png"), (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8))
    shutil.copy(os.path.join(d_hint, "h_0000.png"), os.path.join(work, "hint_first.png"))
    return d_png, d_exr, d_hint


def read_inputs(d, linear):
    """Decode exactly as upstream's read_image_rgb does, into an IMAGE batch."""
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if linear:
            img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(img)
    return torch.from_numpy(np.stack(out))


def read_masks(paths):
    return torch.from_numpy(np.stack([
        cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0 for p in paths]))


def read_exr_dir(d, channels):
    frames = []
    for name in sorted(os.listdir(d)):
        img = cv2.imread(os.path.join(d, name), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if channels == 1:
            frames.append(img if img.ndim == 2 else img[..., 0])
        elif channels == 3:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    return torch.from_numpy(np.stack(frames))


def run_upstream(repo, ffmpeg, inp, out, *extra):
    done = os.path.join(out, ".done")
    if os.path.isfile(done):          # resume: the GPU here is shared, runs can be OOM-killed
        return float(open(done).read() or 0)
    cmd = [sys.executable, os.path.join(repo, "infer.py"), "--input", inp, "--output_dir", out,
           "--outputs", "alpha", "fg", "cutout", "checker", "--output_format", "exr",
           "--exr_dtype", "float", "--hann_chunk", "8", "--hann_stride", "4",
           "--despill_strength", "0.5", "--ffmpeg_bin", ffmpeg,
           "--temp_dir", os.path.join(out, "tmp"), *extra]
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=repo, stdout=subprocess.DEVNULL)
    secs = time.time() - t0
    with open(done, "w") as f:
        f.write(f"{secs:.1f}")
    return secs


def upstream_outputs(out, clip):
    return {k: read_exr_dir(os.path.join(out, f"{clip}_{suffix}_exr"), ch) for k, suffix, ch in (
        ("alpha", "alpha", 1), ("fg", "fg", 3), ("cutout", "cutout", 4),
        ("checker", "checker_comp_despill050", 3))}


def wait_for_vram(gib=7.5, timeout=1800):
    """The test GPU is shared; wait for room rather than OOM someone else's job."""
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free >= gib:
            return
        if time.time() - t0 > timeout:
            raise RuntimeError(f"GPU never had {gib} GiB free (now {free:.1f})")
        print(f"  waiting for VRAM: {free:.1f} GiB free, need {gib}", flush=True)
        time.sleep(15)


def stats(name, ours, ref):
    d = (ours.float() - ref.float()).abs()
    rng = max(float(ref.float().abs().max()), 1e-6)
    line = (f"  {name:<22} max|d| {float(d.max()):.3e}  mean|d| {float(d.mean()):.3e}  "
            f"p99.9 {float(torch.quantile(d.flatten()[::7], 0.999)):.3e}  (ref max {rng:.3f})")
    print(line)
    return float(d.max()), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream_repo", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ffmpeg_bin", default="ffmpeg")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    from ck2 import assets, driver, upstream
    _, _, cu = driver._mods()
    rt = upstream.submodule("runtime")

    d_png, d_exr, d_hint = make_clip(args.work)
    hint_first = read_masks([os.path.join(args.work, "hint_first.png")])
    hint_full = read_masks([os.path.join(d_hint, n) for n in sorted(os.listdir(d_hint))])

    print("upstream infer.py ...")
    up = {}
    secs = {}
    secs["A"] = run_upstream(args.upstream_repo, args.ffmpeg_bin, d_png, os.path.join(args.work, "up_A"),
                             "--hint_first_frame", os.path.join(args.work, "hint_first.png"),
                             "--hint_quality", "0.95")
    up["A"] = upstream_outputs(os.path.join(args.work, "up_A"), "png")
    secs["A2"] = run_upstream(args.upstream_repo, args.ffmpeg_bin, d_png, os.path.join(args.work, "up_A2"),
                              "--hint_first_frame", os.path.join(args.work, "hint_first.png"),
                              "--hint_quality", "0.95")
    up["A2"] = upstream_outputs(os.path.join(args.work, "up_A2"), "png")
    secs["B"] = run_upstream(args.upstream_repo, args.ffmpeg_bin, d_exr, os.path.join(args.work, "up_B"),
                             "--frame_dir_linear", "--hint_video", d_hint, "--hint_quality", "0.75")
    up["B"] = upstream_outputs(os.path.join(args.work, "up_B"), "exr")
    secs["C"] = run_upstream(args.upstream_repo, args.ffmpeg_bin, d_png, os.path.join(args.work, "up_C"),
                             "--hint_first_frame", os.path.join(args.work, "hint_first.png"),
                             "--hint_quality", "0.95", "--low_vram")
    up["C"] = upstream_outputs(os.path.join(args.work, "up_C"), "png")
    print("  upstream seconds:", {k: round(v, 1) for k, v in secs.items()}, "(includes model load)")

    print("pack driver ...")
    model = assets.load_model(allow_download=False)
    base = {"hann_chunk": 8, "hann_stride": 4, "low_vram": "off"}
    imgs_png = read_inputs(d_png, linear=False)
    imgs_exr = read_inputs(d_exr, linear=True)
    ours = {}
    wait_for_vram()
    t0 = time.time()
    ours["A"] = driver.key_clip(model, imgs_png, hint=hint_first, hint_quality=0.95,
                                despill_strength=0.5, settings=base)
    print(f"  A: {time.time() - t0:.1f}s (model already loaded)")
    wait_for_vram()
    ours["B"] = driver.key_clip(model, imgs_exr, hint=hint_full, linear_input=True,
                                hint_quality=0.75, despill_strength=0.5, settings=base)
    wait_for_vram()
    ours["C"] = driver.key_clip(model, imgs_png, hint=hint_first, hint_quality=0.95,
                                despill_strength=0.5, settings=dict(base, low_vram="on"))

    def expected_srgb(u):
        """Pack sRGB-mode outputs, derived from upstream's raw EXRs with upstream's functions."""
        a = u["alpha"].unsqueeze(1)
        fg_lin = u["fg"].permute(0, 3, 1, 2)
        pf = cu.linear_to_srgb(fg_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        fg = rt._despill_average_srgb_torch(pf, strength=0.5)
        checker = torch.from_numpy(upstream.submodule("video_io").make_checker(H, W)).unsqueeze(0)
        comp = cu.srgb_to_linear(fg) * a + cu.srgb_to_linear(checker) * (1 - a)
        prev = cu.linear_to_srgb(comp.clamp(min=0.0)).clamp(0.0, 1.0)
        return fg.permute(0, 2, 3, 1), prev.permute(0, 2, 3, 1)

    print("\nD  upstream vs upstream (GPU noise floor)")
    floor = stats("alpha", up["A2"]["alpha"], up["A"]["alpha"])
    stats("fg (linear)", up["A2"]["fg"], up["A"]["fg"])

    results = {}
    for case in ("A", "C"):
        print(f"\n{case}  sRGB input, first-frame hint{', low-VRAM' if case == 'C' else ''}")
        fg_exp, prev_exp = expected_srgb(up[case])
        results[case] = [stats("alpha", ours[case][0], up[case]["alpha"]),
                         stats("foreground (sRGB)", ours[case][1], fg_exp),
                         stats("preview", ours[case][2], prev_exp)]
    print("\nB  linear EXR input, full hint")
    results["B"] = [stats("alpha", ours["B"][0], up["B"]["alpha"]),
                    stats("foreground (linear)", ours["B"][1], up["B"]["cutout"][..., :3]),
                    stats("preview", ours["B"][2],
                          cu.linear_to_srgb(up["B"]["checker"].clamp(min=0)).clamp(0, 1))]
    print("\nC vs A  (low-VRAM is meant to be exact)")
    stats("alpha", ours["C"][0], ours["A"][0])

    # Pass if every alpha diff is within a few multiples of upstream's own noise floor.
    tol = max(10 * floor[0], 2e-3)
    worst = max(r[0][0] for r in results.values())
    print(f"\nworst alpha max|d| {worst:.3e}  tolerance {tol:.3e}  ->",
          "PASS" if worst <= tol else "FAIL")
    torch.save({k: tuple(x[:, ::4, ::4] for x in v) for k, v in ours.items()},
               os.path.join(args.work, "ours_small.pt"))
    return 0 if worst <= tol else 1


if __name__ == "__main__":
    sys.exit(main())
