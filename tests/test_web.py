"""Run web/bat_corridorkey2.js under quickjs against a stub node (no browser, no ComfyUI).

    pip install --target <dir> --only-binary=:all: quickjs
    PYTHONPATH=<dir> python tests/test_web.py
"""

import json
import os
import re
import sys

import quickjs

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(PACK, "web", "bat_corridorkey2.js"), encoding="utf-8").read()

HARNESS = r"""
const app = { registerExtension(e) { globalThis.__ext = e; } };
%s
function makeNode(mode, preview, hinted) {
    const calls = { cb: 0, created: [] };
    const nodeType = { prototype: { onNodeCreated(...a) { calls.created.push(a.length); } } };
    __ext.beforeRegisterNodeDef(nodeType, { name: "Bat_CorridorKey2_Key" });
    const w = (name, value, type) => ({ name, value, type, computeSize: () => [200, 20],
                                        draw: () => "drawn" });
    const node = Object.create(nodeType.prototype);
    node.size = [300, 200];
    node.inputs = [{ name: "model", link: 1 }, { name: "images", link: 2 },
                   { name: "hint", link: hinted ? 3 : null }, { name: "advanced", link: null }];
    node.widgets = [w("input_colorspace", "sRGB", "combo"), w("hint_quality", 0.95, "number"),
                    w("hint_mode", mode, "combo"), w("hint_frame", 7, "number"),
                    w("preview", preview, "toggle"), w("preview_frame", 12, "number")];
    node.widgets[2].callback = () => { calls.cb++; };
    node.computeSize = () => [250, 30 + node.widgets.reduce((h, x) => h + x.computeSize()[1] + 4, 0)];
    node.setSize = (s) => { node.size = s; };
    node.setDirtyCanvas = () => {};
    node.onNodeCreated(1, 2, 3);
    const byName = n => node.widgets.find(x => x.name === n);
    return { node, calls, byName };
}
function vis(o) {
    const v = {};
    for (const x of o.node.widgets) v[x.name] = !x.hidden;
    v.h = o.node.size[1]; v.cb = o.calls.cb; v.created = o.calls.created;
    v.hint_frame_value = o.byName("hint_frame").value;
    v.hint_frame_draws = o.byName("hint_frame").draw() === "drawn";
    return v;
}
const out = {};
let a = makeNode("all frames", false, false);
out.unhinted = vis(a);
a.node.inputs[2].link = 9; a.node.onConnectionsChange(1, 2, true, {}, {});
out.connected = vis(a);
a.byName("hint_mode").value = "single frame"; a.byName("hint_mode").callback("single frame");
out.single = vis(a);
a.byName("preview").value = true; a.byName("preview").callback(true);
out.preview_on = vis(a);
a.node.inputs[2].link = null; a.node.onConnectionsChange(1, 2, false, {}, {});
out.disconnected = vis(a);
let b = makeNode("all frames", false, false);
b.byName("hint_mode").value = "single frame"; b.byName("preview").value = true;
b.node.inputs[2].link = 5;           // configure() restores values and links after creation
b.node.onConfigure({});
out.loaded = vis(b);
JSON.stringify(out);
"""


def main():
    body = re.sub(r"^\s*import\s.*?;\s*$", "", SRC, flags=re.M | re.S)
    ctx = quickjs.Context()
    out = json.loads(ctx.eval(HARNESS % body))
    fails = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
        if not ok:
            fails.append(name)

    u, c, si, pv, dc, ld = (out[k] for k in ("unhinted", "connected", "single", "preview_on",
                                             "disconnected", "loaded"))
    hint_rows = ("hint_quality", "hint_mode", "hint_frame")
    check("no hint connected: hint rows and preview_frame hidden",
          not any(u[k] for k in hint_rows) and not u["preview_frame"] and u["input_colorspace"], str(u))
    check("onNodeCreated forwards every argument", u["created"] == [3], str(u["created"]))
    check("connecting a hint shows quality and mode, not the frame",
          c["hint_quality"] and c["hint_mode"] and not c["hint_frame"] and c["h"] > u["h"], str(c))
    check("single frame shows hint_frame, drawable again",
          si["hint_frame"] and si["hint_frame_draws"], str(si))
    check("the combo's own callback still runs", si["cb"] == 1)
    check("preview on shows preview_frame", pv["preview_frame"], str(pv))
    check("disconnecting the hint hides all three hint rows",
          not any(dc[k] for k in hint_rows) and dc["preview_frame"], str(dc))
    check("hidden hint_frame keeps its value", dc["hint_frame_value"] == 7)
    check("a saved single-frame + preview workflow loads with its rows visible",
          ld["hint_frame"] and ld["preview_frame"] and ld["hint_mode"], str(ld))
    print(f"\n{len(fails)} failure(s)" if fails else "\nall passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
