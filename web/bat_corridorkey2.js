/**
 * 🦇 CorridorKey 2 — hint_frame only shows when hint_mode is "single frame".
 *
 * Hidden rows keep serialising, so a workflow saved in "all frames" still
 * remembers the frame you last picked. Same hide approach as BAT-NodePack's
 * bat_batch_format.js (type/computeSize/draw/hidden), which drops the row from
 * the layout instead of blanking it.
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
    const mode = node.widgets?.find(w => w.name === "hint_mode");
    const frame = node.widgets?.find(w => w.name === "hint_frame");
    if (!mode || !frame) return;
    setHidden(frame, mode.value !== "single frame");
    // Re-fit, or the node keeps a dead band where the row used to be.
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
            const mode = this.widgets?.find(w => w.name === "hint_mode");
            if (mode) {
                const orig = mode.callback;
                const node = this;
                mode.callback = function (...cbArgs) {
                    const out = orig?.apply(this, cbArgs);
                    syncVisibility(node);
                    return out;
                };
            }
            syncVisibility(this);
            return r;
        };

        // configure() restores widget values after onNodeCreated, so a workflow
        // saved in "single frame" would otherwise load with hint_frame hidden.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const r = onConfigure?.apply(this, args);
            syncVisibility(this);
            return r;
        };
    },
});
