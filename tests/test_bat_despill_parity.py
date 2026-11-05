"""BAT-NodePack's 🦇 Despill must give the same pixels as this pack's key node.

    python tests/test_bat_despill_parity.py

The recommended graph is key node (despill_strength 0) -> 🦇 Despill, so a
despill change re-runs only the cheap node. That is only honest if the two
paths agree. 🦇 Despill is BAT's own MIT code (no Corridor code in it), so this
compares it against upstream's own functions. Skips when BAT-NodePack is not
installed next to this pack. CPU only.
"""

import os
import sys
import types

import numpy as np
import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
sys.path.insert(1, os.path.dirname(os.path.dirname(PACK)))
BAT = os.path.join(os.path.dirname(PACK), "ComfyUI-BAT-NodePack")
if not os.path.isfile(os.path.join(BAT, "bat_despill.py")):
    print("SKIP: ComfyUI-BAT-NodePack (with bat_despill.py) is not installed next to this pack")
    sys.exit(0)

from ck2 import driver  # noqa: E402

pkg = types.ModuleType("batpack")
pkg.__path__ = [BAT]
sys.modules["batpack"] = pkg
from batpack import bat_despill as bd  # noqa: E402

rt, vio, cu = driver._mods()
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


rng = np.random.default_rng(2)
T, H, W = 3, 150, 230
fg = torch.from_numpy(rng.uniform(0, 1, (T, H, W, 3)).astype(np.float32))
fg[..., 1] = (fg[..., 1] + 0.2).clamp(0, 1)
alpha = torch.from_numpy(rng.uniform(0, 1, (T, H, W)).astype(np.float32))
chw = lambda x: x.permute(0, 3, 1, 2)

x = torch.from_numpy(rng.uniform(-0.5, 4.0, 20000).astype(np.float32))
d = max((bd.srgb_to_linear(x) - cu.srgb_to_linear(x)).abs().max().item(),
        (bd.linear_to_srgb(x) - cu.linear_to_srgb(x)).abs().max().item())
check("sRGB transfer == upstream color_utils (incl. negatives, > 1)", d < 1e-6, f"max |d| {d:.1e}")

for screen in ("green", "blue"):
    for s in (0.0, 0.3, 0.5, 1.0):
        ours = chw(bd.despill(fg, s, screen))
        theirs = driver.despill(chw(fg), s, screen)
        d = (ours - theirs).abs().max().item()
        check(f"{screen} despill {s} == key node's driver.despill", d < 1e-6, f"max |d| {d:.1e}")

# key at despill s  ==  key at despill 0 -> 🦇 Despill at s
key_s = driver.despill(chw(fg), 0.7, "green")
two = chw(bd.BatDespill().run(driver.despill(chw(fg), 0.0, "green").permute(0, 2, 3, 1), alpha,
                              "sRGB", "green", 0.7, "checker")["result"][0])
check("key(despill .7) == 🦇 Despill(key(despill 0), .7)", (key_s - two).abs().max().item() < 1e-6)

comp = bd.BatDespill().run(fg, alpha, "sRGB", "green", 0.5, "checker")["result"][2][1]
clean = bd.despill(fg[1], 0.5, "green")
up = vio.composite_over_checker_srgb(clean.permute(2, 0, 1).numpy(), alpha[1][None].numpy(), vio.make_checker(H, W))
d = np.abs(comp.permute(2, 0, 1).numpy() - up).max()
check("🦇 Despill checker comp == key node's preview (upstream composite_over_checker_srgb)", d < 1e-5, f"max |d| {d:.1e}")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
