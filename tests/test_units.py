"""CPU-only checks: no GPU, no weights, no ComfyUI server.

    python tests/test_units.py

Needs the upstream runtime fetched (ck2/upstream.py) and ComfyUI's folder_paths
importable (run from inside a ComfyUI install).
"""

import inspect
import os
import sys
import tempfile

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
sys.path.insert(1, os.path.dirname(os.path.dirname(PACK)))

from ck2 import driver, nodes, upstream  # noqa: E402

rt, vio, cu = driver._mods()
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def test_preprocess_matches_upstream_readers():
    rng = np.random.default_rng(0)
    h, w = 75, 130                                     # pads to 128 x 256
    with tempfile.TemporaryDirectory() as d:
        u8 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(d, "a.png"), u8)
        lin = rng.uniform(-0.05, 4.0, (h, w, 3)).astype(np.float32)
        cv2.imwrite(os.path.join(d, "a.exr"), lin, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        for ext, linear in (("png", False), ("exr", True)):
            path = os.path.join(d, f"a.{ext}")
            meta = driver.build_meta(1, h, w, driver.DEFAULTS)
            meta["exr_linear"] = linear
            ref = vio.preprocess_frame_dir_path(path, meta, need_model=True)
            img = vio.read_image_rgb(path)
            ours = driver.FrameSource(torch.from_numpy(img[None]), meta, linear).views(0)
            check(f"frame views == upstream frame-dir reader ({ext})",
                  all(same(o, r) for o, r in zip(ours, ref)))

        m = rng.uniform(0, 1, (h, w)).astype(np.float32)
        hp = os.path.join(d, "h.exr")
        cv2.imwrite(hp, m, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        meta = driver.build_meta(1, h, w, driver.DEFAULTS)
        ref_first = vio.load_initial_hint_frame(hp, meta)
        ref_full = vio.preprocess_hint_frame(cv2.imread(hp, cv2.IMREAD_UNCHANGED), meta)
        ours = driver.HintSource(torch.from_numpy(m[None]), meta).frame(0)
        check("hint == upstream first-frame hint", same(ours, ref_first))
        check("hint == upstream per-frame hint", same(ours, ref_full))


def test_linear_adjust_follows_video_path():
    rng = np.random.default_rng(1)
    h, w = 64, 64
    bgr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    s = dict(driver.DEFAULTS, linear_brightness=1.7, linear_contrast=0.8)
    meta = driver.build_meta(1, h, w, s)
    ref = vio.preprocess_video_frame_views(bgr, meta, need_model=True)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ours = driver.FrameSource(torch.from_numpy(img[None]), meta, False).views(0)
    check("brightness/contrast == upstream video-path views", all(same(o, r) for o, r in zip(ours, ref)))


def test_despill():
    x = torch.rand(2, 3, 8, 8) * 1.5
    g = driver.despill(x, 0.7, "green")
    check("green despill is upstream's", torch.equal(g, rt._despill_average_srgb_torch(x, 0.7)))
    swap = [0, 2, 1]                                  # exchange G and B
    b = driver.despill(x[:, swap], 0.7, "blue")[:, swap]
    check("blue despill mirrors green", torch.allclose(b, g, atol=0))
    check("despill 0 is identity", torch.equal(driver.despill(x, 0.0, "blue"), x))


def test_profiles():
    for hw in ((1152, 1920), (2176, 3840), (512, 768)):
        for T in (1, 30, 400):
            for budget in (6.0, 12.0, 24.0, 48.0, 96.0):
                check(f"auto profile == upstream {hw} T={T} {budget}GB",
                      driver.pick_profile(hw, T, budget) == rt.select_vram_profile(hw, T, budget))
    c, s, lv, _ = driver.pick_profile((2176, 3840), 100, 24.0, 48, "auto")
    check("pinned 4K window 48 on 24 GB falls back to low-VRAM", (c, s, lv) == (48, 24, True))
    c, s, lv, _ = driver.pick_profile((1152, 1920), 100, 24.0, 0, "off")
    check("low_vram=off picks a normal-mode window", lv is False and c in rt._VRAM_PEAK_GB[False][0])


def test_build_env_isolation():
    foreign = type(sys)("moge")
    foreign.__file__ = "/elsewhere/moge/__init__.py"
    sys.modules["moge"] = foreign
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    with upstream.isolated_build_env():
        check("offline flags set while building",
              os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1")
        import moge.model.v2 as ours
        check("our MoGe imported, not the foreign one", ours.__file__.startswith(upstream.UPSTREAM_DIR))
    check("HF_HUB_OFFLINE removed again", "HF_HUB_OFFLINE" not in os.environ)
    check("TRANSFORMERS_OFFLINE restored", os.environ.get("TRANSFORMERS_OFFLINE") == "0")
    check("foreign moge restored", sys.modules.get("moge") is foreign)
    check("no moge.* of ours left behind",
          not any(k.startswith("moge.") for k in sys.modules))
    del sys.modules["moge"]


def test_manifest():
    check("upstream tree verifies", upstream.verify() == [])
    check("manifest pins a full commit", len(upstream.COMMIT) == 40)


def test_nodes():
    adv = nodes.CorridorKey2AdvancedParams
    req = adv.INPUT_TYPES()["required"]
    built = adv().build(**{k: v[1]["default"] for k, v in req.items()})[0]
    check("advanced node defaults == driver.DEFAULTS", built == driver.DEFAULTS,
          str({k: (built.get(k), driver.DEFAULTS.get(k)) for k in set(built) | set(driver.DEFAULTS)
               if built.get(k) != driver.DEFAULTS.get(k)}))
    for cls in (nodes.CorridorKey2, nodes.CorridorKey2AdvancedParams, nodes.CorridorKey2LoadModel):
        spec = cls.INPUT_TYPES()
        names = list(spec["required"]) + list(spec.get("optional", {}))
        params = list(inspect.signature(getattr(cls, cls.FUNCTION)).parameters)[1:]
        accepts_kw = any(p.kind == p.VAR_KEYWORD for p in
                         inspect.signature(getattr(cls, cls.FUNCTION)).parameters.values())
        check(f"{cls.__name__} inputs match its function",
              accepts_kw and set(params[:2]) <= set(names) or params == names, f"{names} vs {params}")
    check("key node widget order is the saved-workflow contract",
          [k for k, v in nodes.CorridorKey2.INPUT_TYPES()["required"].items()
           if isinstance(v[0], list) or v[0] in ("FLOAT", "INT", "BOOLEAN", "STRING")]
          == ["input_colorspace", "screen", "hint_quality", "despill_strength"])


def test_hint_validation():
    imgs = torch.zeros(5, 64, 64, 3)
    try:
        driver.key_clip(None, imgs, hint=torch.zeros(3, 64, 64))
        check("3 hint frames for 5 images is rejected", False)
    except ValueError as e:
        check("3 hint frames for 5 images is rejected", "one mask per frame" in str(e))


for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
    fn()
print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
