/**
 * 🦇 CorridorKey 2 — show only the controls that do something.
 *
 *   hint_quality, hint_mode  only while a mask is connected to `hint`
 *   hint_frame               ... and hint_mode is "single frame"
 *   preview_frame            only while `preview` is on
 *
 * Hidden rows keep serialising, so values survive being hidden. Same hide
 * approach as BAT-NodePack's bat_batch_format.js (type/computeSize/draw/hidden),
 * which drops the row from the layout instead of blanking it.
 */

import { app } from "/scripts/app.js";

const NODE_TYPE = "Bat_CorridorKey2_Key";

function setHidden(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (widget._ck2OrigType === undefined) {
            widget._ck2OrigType = widget.type;
            widget._ck2OrigComputeSize = widget.computeSize;
            widget._ck2OrigDraw = widget.draw;
        }
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        widget.hidden = true;
    } else if (widget._ck2OrigType !== undefined) {
        widget.type = widget._ck2OrigType;
        widget.computeSize = widget._ck2OrigComputeSize;
        widget.draw = widget._ck2OrigDraw;
        widget.hidden = false;
    }
}

function syncVisibility(node) {
    const get = name => node.widgets?.find(w => w.name === name);
    const mode = get("hint_mode");
    const preview = get("preview");
    if (!mode || !preview) return;
    const hinted = node.inputs?.find(i => i.name === "hint")?.link != null;
    setHidden(get("hint_quality"), !hinted);
    setHidden(mode, !hinted);
    setHidden(get("hint_frame"), !hinted || mode.value !== "single frame");
    setHidden(get("preview_frame"), !preview.value);
    // Re-fit, or the node keeps a dead band where the rows used to be.
    const min = node.computeSize?.();
    if (min) node.setSize([Math.max(node.size[0], min[0]), min[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Bat.CorridorKey2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const r = onNodeCreated?.apply(this, args);
            const node = this;
            for (const name of ["hint_mode", "preview"]) {
                const widget = this.widgets?.find(w => w.name === name);
                if (!widget) continue;
                const orig = widget.callback;
                widget.callback = function (...cbArgs) {
                    const out = orig?.apply(this, cbArgs);
                    syncVisibility(node);
                    return out;
                };
            }
            syncVisibility(this);
            return r;
        };

        // configure() restores widget values and input links after onNodeCreated.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const r = onConfigure?.apply(this, args);
            syncVisibility(this);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const r = onConnectionsChange?.apply(this, args);
            syncVisibility(this);
            return r;
        };
    },
});
