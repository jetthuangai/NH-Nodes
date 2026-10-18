// Helpers shared by the NH-Nodes frontend extensions. This file registers no extension and imports nothing: ComfyUI
// loads every .js file of the web folder, the extensions import this one with a relative path, and the browser keeps a
// single copy of it.

export function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget.name === name);
}

export function toInt(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
}

// `value` as an integer between min and max; `fallback` (also clamped) when it is not a number.
export function clampInt(value, min, max, fallback) {
    return Math.max(min, Math.min(max, toInt(value, fallback)));
}

// Hiding a widget.
//
// ComfyUI frontend 1.55.17 (litegraph BaseWidget) has `widget.hidden`: a hidden widget is not drawn, takes no row, cannot
// be clicked, and is still saved with the workflow and sent with the prompt. Older frontends have no such property, so
// there a widget is hidden the way this pack always did it: a type no renderer knows and a computeSize that takes no
// room. `serialize` is never touched, so the value of a hidden widget is saved either way.
//
// showWidget only undoes what hideWidget did: a widget the node definition hides stays hidden.
const HIDDEN_STATE = "__nhHiddenState";
const HIDDEN_TYPE = "nh-hidden";

export function hideWidget(widget) {
    if (!widget || widget[HIDDEN_STATE]) {
        return;
    }

    if ("hidden" in widget) {
        widget[HIDDEN_STATE] = { native: true };
        widget.hidden = true;
        return;
    }

    widget[HIDDEN_STATE] = {
        native: false,
        type: widget.type,
        computeSize: widget.computeSize,
        ownComputeSize: Object.prototype.hasOwnProperty.call(widget, "computeSize"),
    };
    widget.computeSize = () => [0, -4];
    widget.type = HIDDEN_TYPE;
}

export function showWidget(widget) {
    const state = widget?.[HIDDEN_STATE];
    if (!state) {
        return;
    }
    delete widget[HIDDEN_STATE];

    if (state.native) {
        widget.hidden = false;
        return;
    }

    widget.type = state.type;
    if (state.ownComputeSize) {
        widget.computeSize = state.computeSize;
    } else {
        delete widget.computeSize;
    }
}

// Calls `after` each time the widget's callback fires, once the callback that was already there (the frontend's, or
// another extension's) has run. `after` gets the same `this` and arguments.
export function chainCallback(widget, after) {
    const original = widget.callback;
    widget.callback = function () {
        original?.apply(this, arguments);
        after.apply(this, arguments);
    };
}

// Repaints the canvas. `graph` is the application's graph (app.graph), which can differ from node.graph.
export function markDirty(node, graph) {
    node.setDirtyCanvas?.(true, true);
    graph?.setDirtyCanvas?.(true, true);
}

// Fits the node to the widgets and slots it has now, then repaints.
export function refreshNode(node, graph) {
    node.setSize?.(node.computeSize?.() || node.size);
    markDirty(node, graph);
}
