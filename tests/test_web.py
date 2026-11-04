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
function makeNode(mode) {
    const calls = { cb: 0, created: [] };
    const nodeType = { prototype: { onNodeCreated(...a) { calls.created.push(a.length); } } };
    __ext.beforeRegisterNodeDef(nodeType, { name: "Bat_CorridorKey2_Key" });
    const w = (name, value, type) => ({ name, value, type, computeSize: () => [200, 20],
                                        draw: () => "drawn" });
    const node = Object.create(nodeType.prototype);
    node.size = [300, 200];
    node.widgets = [w("hint_quality", 0.95, "number"), w("hint_mode", mode, "combo"),
                    w("hint_frame", 7, "number")];
    node.widgets[1].callback = () => { calls.cb++; };
    node.computeSize = () => [250, 30 + node.widgets.reduce((h, x) => h + x.computeSize()[1] + 4, 0)];
    node.setSize = (s) => { node.size = s; };
    node.setDirtyCanvas = () => {};
    node.onNodeCreated(1, 2, 3);
    return { node, calls, frame: node.widgets[2], mode: node.widgets[1] };
}
function state(o) {
    return { hidden: !!o.frame.hidden, type: o.frame.type, h: o.node.size[1],
             value: o.frame.value, draws: o.frame.draw() === "drawn", cb: o.calls.cb,
             created: o.calls.created };
}
const out = {};
let a = makeNode("all frames");
out.created_all = state(a);
a.mode.value = "single frame"; a.mode.callback("single frame");
out.switched_single = state(a);
a.mode.value = "all frames"; a.mode.callback("all frames");
out.switched_back = state(a);
let b = makeNode("all frames");
b.mode.value = "single frame";          // configure() writes saved values after creation
b.node.onConfigure({});
out.loaded_single = state(b);
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

    c, s, back, load = out["created_all"], out["switched_single"], out["switched_back"], out["loaded_single"]
    check("all frames: hint_frame hidden on creation", c["hidden"] and c["type"] == "hidden", str(c))
    check("onNodeCreated forwards every argument", c["created"] == [3], str(c["created"]))
    check("switching to single frame shows it again",
          not s["hidden"] and s["type"] == "number" and s["draws"], str(s))
    check("the node grows back by one row", s["h"] > c["h"], f"{c['h']} -> {s['h']}")
    check("the combo's own callback still runs", s["cb"] == 1 and back["cb"] == 2)
    check("switching back hides it and shrinks the node", back["hidden"] and back["h"] == c["h"])
    check("hidden hint_frame keeps its value", back["value"] == 7)
    check("a workflow saved in single frame loads with hint_frame visible", not load["hidden"], str(load))
    print(f"\n{len(fails)} failure(s)" if fails else "\nall passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
