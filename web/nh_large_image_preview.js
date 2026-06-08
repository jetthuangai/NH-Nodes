import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREVIEW_NODE_NAME = "NH_LargeImagePreview";
const COMPARE_NODE_NAME = "NH_LargeImageCompare";

function ensurePreviewButton(node) {
    if (node.__nhLargePreviewButton) {
        return;
    }

    node.__nhLargePreviewButton = node.addWidget("button", "Open Viewer", "Open Viewer", () => {
        const manifest = node.__nhLargePreviewManifest?.[0];
        if (!manifest) {
            alert("NH Large Image Preview has no viewer data yet. Queue the node first.");
            return;
        }
        openManifest(manifest);
    });
}

function ensureCompareButton(node) {
    if (node.__nhLargeCompareButton) {
        return;
    }

    node.__nhLargeCompareButton = node.addWidget("button", "Open Compare", "Open Compare", () => {
        const manifest = node.__nhLargeCompareManifest?.[0];
        if (!manifest) {
            alert("NH Large Image Compare has no viewer data yet. Queue the node first.");
            return;
        }
        openCompareManifest(manifest);
    });
}

function apiUrl(path) {
    return api.apiURL(`${path}${path.includes("?") ? "&" : "?"}r=${Date.now()}`);
}

function thumbUrl(manifest) {
    if (manifest?.thumb_url) {
        return apiUrl(manifest.thumb_url);
    }
    const thumb = manifest?.thumb;
    if (!thumb) {
        return "";
    }
    const params = new URLSearchParams({
        filename: thumb.filename,
        type: thumb.type || "temp",
        subfolder: thumb.subfolder || "",
        r: `${Date.now()}`,
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function tileUrl(manifest, level, x, y) {
    const template = manifest.tile_url_template || "";
    return api.apiURL(template
        .replace("{level}", `${level}`)
        .replace("{x}", `${x}`)
        .replace("{y}", `${y}`));
}

function openManifest(manifest) {
    if (!manifest?.levels?.length) {
        const url = thumbUrl(manifest);
        if (url) {
            window.open(url, "_blank", "noopener,noreferrer");
            return;
        }
        alert("NH Large Image Preview has no thumbnail or tile data.");
        return;
    }
    openTiledViewer(manifest);
}

function openCompareManifest(compare) {
    if (!compare?.a || !compare?.b) {
        alert("NH Large Image Compare has no A/B tile data.");
        return;
    }
    openCompareViewer(compare);
}

function makeButton(label, onClick) {
    const button = document.createElement("button");
    button.textContent = label;
    button.onclick = onClick;
    button.style.cssText = [
        "border:1px solid rgba(255,255,255,0.18)",
        "background:#2d2d2d",
        "color:#fff",
        "border-radius:6px",
        "padding:6px 10px",
        "cursor:pointer",
        "font:12px system-ui,sans-serif",
    ].join(";");
    return button;
}

function openTiledViewer(manifest) {
    const overlay = document.createElement("div");
    let closeOnEscape;
    let onMouseMove;
    let onMouseUp;
    const closeViewer = () => {
        overlay.remove();
        if (closeOnEscape) {
            document.removeEventListener("keydown", closeOnEscape);
        }
        if (onMouseMove) {
            window.removeEventListener("mousemove", onMouseMove);
        }
        if (onMouseUp) {
            window.removeEventListener("mouseup", onMouseUp);
        }
        window.removeEventListener("resize", resizeCanvas);
    };

    overlay.style.cssText = [
        "position:fixed",
        "inset:0",
        "z-index:100000",
        "background:rgba(0,0,0,0.82)",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "padding:18px",
    ].join(";");

    const panel = document.createElement("div");
    panel.style.cssText = [
        "width:min(96vw,1500px)",
        "height:min(94vh,980px)",
        "display:flex",
        "flex-direction:column",
        "gap:10px",
        "background:#171717",
        "border:1px solid rgba(255,255,255,0.16)",
        "border-radius:8px",
        "padding:12px",
        "box-shadow:0 24px 80px rgba(0,0,0,0.5)",
    ].join(";");

    const header = document.createElement("div");
    header.style.cssText = [
        "display:flex",
        "align-items:center",
        "justify-content:space-between",
        "gap:16px",
        "color:#eee",
        "font:13px system-ui,sans-serif",
        "min-height:32px",
    ].join(";");

    const title = document.createElement("div");
    title.textContent = `${manifest.title || "NH Large Image Preview"} | ${manifest.width}x${manifest.height} | ${manifest.levels.length} levels`;
    title.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap";

    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end";

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = [
        "position:relative",
        "flex:1 1 auto",
        "min-height:300px",
        "overflow:hidden",
        "background:#101010",
        "border:1px solid rgba(255,255,255,0.12)",
        "border-radius:4px",
        "cursor:grab",
    ].join(";");

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;height:100%";
    canvasWrap.appendChild(canvas);

    const status = document.createElement("div");
    status.style.cssText = [
        "position:absolute",
        "left:10px",
        "bottom:8px",
        "padding:4px 7px",
        "border-radius:5px",
        "background:rgba(0,0,0,0.58)",
        "color:#ddd",
        "font:12px system-ui,sans-serif",
        "pointer-events:none",
    ].join(";");
    canvasWrap.appendChild(status);

    panel.appendChild(header);
    panel.appendChild(canvasWrap);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    const ctx = canvas.getContext("2d");
    const levels = [...manifest.levels].sort((a, b) => a.scale - b.scale);
    const fullLevel = levels[levels.length - 1];
    const tileCache = new Map();
    const state = {
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
        dragging: false,
        lastX: 0,
        lastY: 0,
        raf: 0,
    };

    function getCanvasSize() {
        return {
            width: Math.max(1, canvas.clientWidth || canvasWrap.clientWidth || 1),
            height: Math.max(1, canvas.clientHeight || canvasWrap.clientHeight || 1),
        };
    }

    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const size = getCanvasSize();
        canvas.width = Math.max(1, Math.floor(size.width * dpr));
        canvas.height = Math.max(1, Math.floor(size.height * dpr));
        scheduleDraw();
    }

    function fit() {
        const size = getCanvasSize();
        state.zoom = Math.min(size.width / manifest.width, size.height / manifest.height);
        state.offsetX = (size.width - manifest.width * state.zoom) / 2;
        state.offsetY = (size.height - manifest.height * state.zoom) / 2;
        scheduleDraw();
    }

    function zoomAt(x, y, factor) {
        const beforeX = (x - state.offsetX) / state.zoom;
        const beforeY = (y - state.offsetY) / state.zoom;
        const fitZoom = Math.min(canvas.clientWidth / manifest.width, canvas.clientHeight / manifest.height);
        state.zoom = Math.max(fitZoom * 0.25, Math.min(4, state.zoom * factor));
        state.offsetX = x - beforeX * state.zoom;
        state.offsetY = y - beforeY * state.zoom;
        scheduleDraw();
    }

    function setHundredPercent() {
        const size = getCanvasSize();
        state.zoom = 1;
        state.offsetX = (size.width - manifest.width) / 2;
        state.offsetY = (size.height - manifest.height) / 2;
        scheduleDraw();
    }

    function chooseLevel() {
        const desired = state.zoom * (window.devicePixelRatio || 1);
        for (const level of levels) {
            if (level.scale >= desired) {
                return level;
            }
        }
        return fullLevel;
    }

    function getTile(level, x, y) {
        const key = `${level.level}:${x}:${y}`;
        let item = tileCache.get(key);
        if (item) {
            return item;
        }

        const image = new Image();
        item = { image, loaded: false, failed: false };
        tileCache.set(key, item);
        image.onload = () => {
            item.loaded = true;
            scheduleDraw();
        };
        image.onerror = () => {
            item.failed = true;
            scheduleDraw();
        };
        image.src = tileUrl(manifest, level.level, x, y);
        return item;
    }

    function draw() {
        state.raf = 0;
        const dpr = window.devicePixelRatio || 1;
        const size = getCanvasSize();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, size.width, size.height);
        ctx.fillStyle = "#101010";
        ctx.fillRect(0, 0, size.width, size.height);

        const level = chooseLevel();
        const scale = level.scale || 1;
        const tileSize = level.tile_size || manifest.tile_size || 512;
        const sourceLeft = Math.max(0, (0 - state.offsetX) / state.zoom);
        const sourceTop = Math.max(0, (0 - state.offsetY) / state.zoom);
        const sourceRight = Math.min(manifest.width, (size.width - state.offsetX) / state.zoom);
        const sourceBottom = Math.min(manifest.height, (size.height - state.offsetY) / state.zoom);

        const tileLeft = Math.max(0, Math.floor((sourceLeft * scale) / tileSize));
        const tileTop = Math.max(0, Math.floor((sourceTop * scale) / tileSize));
        const tileRight = Math.min(level.columns - 1, Math.floor((sourceRight * scale) / tileSize));
        const tileBottom = Math.min(level.rows - 1, Math.floor((sourceBottom * scale) / tileSize));

        let visible = 0;
        let loaded = 0;
        for (let ty = tileTop; ty <= tileBottom; ty++) {
            for (let tx = tileLeft; tx <= tileRight; tx++) {
                visible += 1;
                const tile = getTile(level, tx, ty);
                const levelX = tx * tileSize;
                const levelY = ty * tileSize;
                const sourceX = levelX / scale;
                const sourceY = levelY / scale;
                const drawX = sourceX * state.zoom + state.offsetX;
                const drawY = sourceY * state.zoom + state.offsetY;

                if (tile.loaded) {
                    loaded += 1;
                    const drawW = (tile.image.naturalWidth / scale) * state.zoom;
                    const drawH = (tile.image.naturalHeight / scale) * state.zoom;
                    ctx.drawImage(tile.image, drawX, drawY, drawW, drawH);
                } else if (!tile.failed) {
                    const fallbackW = Math.min(tileSize, level.width - levelX) / scale * state.zoom;
                    const fallbackH = Math.min(tileSize, level.height - levelY) / scale * state.zoom;
                    ctx.fillStyle = "#181818";
                    ctx.fillRect(drawX, drawY, fallbackW, fallbackH);
                }
            }
        }

        ctx.strokeStyle = "rgba(255,255,255,0.22)";
        ctx.lineWidth = 1;
        ctx.strokeRect(state.offsetX, state.offsetY, manifest.width * state.zoom, manifest.height * state.zoom);
        status.textContent = `${Math.round(state.zoom * 100)}% | level ${level.level} (${level.width}x${level.height}) | tiles ${loaded}/${visible}`;
    }

    function scheduleDraw() {
        if (!state.raf) {
            state.raf = requestAnimationFrame(draw);
        }
    }

    buttons.appendChild(makeButton("Fit", fit));
    buttons.appendChild(makeButton("100%", setHundredPercent));
    buttons.appendChild(makeButton("-", () => zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 0.8)));
    buttons.appendChild(makeButton("+", () => zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 1.25)));
    buttons.appendChild(makeButton("Thumb", () => window.open(thumbUrl(manifest), "_blank", "noopener,noreferrer")));
    buttons.appendChild(makeButton("Close", closeViewer));
    header.appendChild(title);
    header.appendChild(buttons);

    canvasWrap.addEventListener("mousedown", (event) => {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvasWrap.style.cursor = "grabbing";
    });
    onMouseMove = (event) => {
        if (!state.dragging) {
            return;
        }
        state.offsetX += event.clientX - state.lastX;
        state.offsetY += event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        scheduleDraw();
    };
    onMouseUp = () => {
        state.dragging = false;
        canvasWrap.style.cursor = "grab";
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    canvasWrap.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        zoomAt(x, y, event.deltaY < 0 ? 1.18 : 0.85);
    }, { passive: false });

    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            closeViewer();
        }
    });
    closeOnEscape = function (event) {
        if (event.key === "Escape" && document.body.contains(overlay)) {
            closeViewer();
        }
    };
    document.addEventListener("keydown", closeOnEscape);
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    fit();
}

function openCompareViewer(compare) {
    const manifestA = compare.a;
    const manifestB = compare.b;
    const baseWidth = Math.max(manifestA.width || 1, manifestB.width || 1);
    const baseHeight = Math.max(manifestA.height || 1, manifestB.height || 1);

    const overlay = document.createElement("div");
    let closeOnEscape;
    let onMouseMove;
    let onMouseUp;
    const closeViewer = () => {
        overlay.remove();
        if (closeOnEscape) {
            document.removeEventListener("keydown", closeOnEscape);
        }
        if (onMouseMove) {
            window.removeEventListener("mousemove", onMouseMove);
        }
        if (onMouseUp) {
            window.removeEventListener("mouseup", onMouseUp);
        }
        window.removeEventListener("resize", resizeCanvas);
    };

    overlay.style.cssText = [
        "position:fixed",
        "inset:0",
        "z-index:100000",
        "background:rgba(0,0,0,0.84)",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "padding:18px",
    ].join(";");

    const panel = document.createElement("div");
    panel.style.cssText = [
        "width:min(96vw,1500px)",
        "height:min(94vh,980px)",
        "display:flex",
        "flex-direction:column",
        "gap:10px",
        "background:#171717",
        "border:1px solid rgba(255,255,255,0.16)",
        "border-radius:8px",
        "padding:12px",
        "box-shadow:0 24px 80px rgba(0,0,0,0.5)",
    ].join(";");

    const header = document.createElement("div");
    header.style.cssText = [
        "display:flex",
        "align-items:center",
        "justify-content:space-between",
        "gap:16px",
        "color:#eee",
        "font:13px system-ui,sans-serif",
        "min-height:32px",
    ].join(";");

    const title = document.createElement("div");
    title.textContent = `${compare.title || "NH Large Image Compare"} | A ${manifestA.width}x${manifestA.height} | B ${manifestB.width}x${manifestB.height}`;
    title.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap";

    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end";

    const splitInput = document.createElement("input");
    splitInput.type = "range";
    splitInput.min = "5";
    splitInput.max = "95";
    splitInput.value = "50";
    splitInput.style.cssText = "width:110px;accent-color:#7ab7ff";

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = [
        "position:relative",
        "flex:1 1 auto",
        "min-height:300px",
        "overflow:hidden",
        "background:#101010",
        "border:1px solid rgba(255,255,255,0.12)",
        "border-radius:4px",
        "cursor:grab",
    ].join(";");

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;height:100%";
    canvasWrap.appendChild(canvas);

    const status = document.createElement("div");
    status.style.cssText = [
        "position:absolute",
        "left:10px",
        "bottom:8px",
        "padding:4px 7px",
        "border-radius:5px",
        "background:rgba(0,0,0,0.58)",
        "color:#ddd",
        "font:12px system-ui,sans-serif",
        "pointer-events:none",
    ].join(";");
    canvasWrap.appendChild(status);

    panel.appendChild(header);
    panel.appendChild(canvasWrap);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    const ctx = canvas.getContext("2d");
    const levelSets = new Map();
    const frameLayouts = new Map();
    const tileCache = new Map();
    const state = {
        mode: compare.compare_mode || "slider",
        split: 0.5,
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
        dragging: false,
        lastX: 0,
        lastY: 0,
        raf: 0,
    };

    function getCanvasSize() {
        return {
            width: Math.max(1, canvas.clientWidth || canvasWrap.clientWidth || 1),
            height: Math.max(1, canvas.clientHeight || canvasWrap.clientHeight || 1),
        };
    }

    function getFrame(manifest) {
        const key = manifest.cache_id;
        let frame = frameLayouts.get(key);
        if (!frame) {
            const sourceWidth = Math.max(1, manifest.width || 1);
            const sourceHeight = Math.max(1, manifest.height || 1);
            const scale = Math.min(baseWidth / sourceWidth, baseHeight / sourceHeight);
            const width = sourceWidth * scale;
            const height = sourceHeight * scale;
            frame = {
                scale,
                width,
                height,
                padX: (baseWidth - width) / 2,
                padY: (baseHeight - height) / 2,
            };
            frameLayouts.set(key, frame);
        }
        return frame;
    }

    function getLevelSet(manifest) {
        const key = manifest.cache_id;
        let set = levelSets.get(key);
        if (!set) {
            const levels = [...(manifest.levels || [])].sort((a, b) => a.scale - b.scale);
            set = { levels, fullLevel: levels[levels.length - 1] };
            levelSets.set(key, set);
        }
        return set;
    }

    function chooseLevel(manifest) {
        const set = getLevelSet(manifest);
        const desired = state.zoom * getFrame(manifest).scale * (window.devicePixelRatio || 1);
        for (const level of set.levels) {
            if (level.scale >= desired) {
                return level;
            }
        }
        return set.fullLevel;
    }

    function getTile(manifest, level, x, y) {
        const key = `${manifest.cache_id}:${level.level}:${x}:${y}`;
        let item = tileCache.get(key);
        if (item) {
            return item;
        }

        const image = new Image();
        item = { image, loaded: false, failed: false };
        tileCache.set(key, item);
        image.onload = () => {
            item.loaded = true;
            scheduleDraw();
        };
        image.onerror = () => {
            item.failed = true;
            scheduleDraw();
        };
        image.src = tileUrl(manifest, level.level, x, y);
        return item;
    }

    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const size = getCanvasSize();
        canvas.width = Math.max(1, Math.floor(size.width * dpr));
        canvas.height = Math.max(1, Math.floor(size.height * dpr));
        scheduleDraw();
    }

    function activePaneWidth() {
        const size = getCanvasSize();
        return state.mode === "side_by_side" ? size.width / 2 : size.width;
    }

    function fit() {
        const size = getCanvasSize();
        const width = activePaneWidth();
        state.zoom = Math.min(width / baseWidth, size.height / baseHeight);
        state.offsetX = (width - baseWidth * state.zoom) / 2;
        state.offsetY = (size.height - baseHeight * state.zoom) / 2;
        scheduleDraw();
    }

    function localXForMode(x) {
        const size = getCanvasSize();
        if (state.mode === "side_by_side" && x > size.width / 2) {
            return x - size.width / 2;
        }
        return x;
    }

    function zoomAt(x, y, factor) {
        const localX = localXForMode(x);
        const beforeX = (localX - state.offsetX) / state.zoom;
        const beforeY = (y - state.offsetY) / state.zoom;
        const fitZoom = Math.min(activePaneWidth() / baseWidth, canvas.clientHeight / baseHeight);
        state.zoom = Math.max(fitZoom * 0.25, Math.min(4, state.zoom * factor));
        state.offsetX = localX - beforeX * state.zoom;
        state.offsetY = y - beforeY * state.zoom;
        scheduleDraw();
    }

    function setHundredPercent() {
        const size = getCanvasSize();
        const width = activePaneWidth();
        state.zoom = 1;
        state.offsetX = (width - baseWidth) / 2;
        state.offsetY = (size.height - baseHeight) / 2;
        scheduleDraw();
    }

    function drawTiles(manifest, drawOffsetX, drawOffsetY, viewportX, viewportY, viewportW, viewportH) {
        const level = chooseLevel(manifest);
        if (!level) {
            return { visible: 0, loaded: 0, level: null };
        }

        const frame = getFrame(manifest);
        const levelScale = level.scale || 1;
        const tileSize = level.tile_size || manifest.tile_size || 512;
        const logicalLeft = (viewportX - drawOffsetX) / state.zoom;
        const logicalTop = (viewportY - drawOffsetY) / state.zoom;
        const logicalRight = (viewportX + viewportW - drawOffsetX) / state.zoom;
        const logicalBottom = (viewportY + viewportH - drawOffsetY) / state.zoom;
        const sourceLeft = Math.max(0, Math.min(manifest.width, (logicalLeft - frame.padX) / frame.scale));
        const sourceTop = Math.max(0, Math.min(manifest.height, (logicalTop - frame.padY) / frame.scale));
        const sourceRight = Math.max(0, Math.min(manifest.width, (logicalRight - frame.padX) / frame.scale));
        const sourceBottom = Math.max(0, Math.min(manifest.height, (logicalBottom - frame.padY) / frame.scale));
        if (sourceRight <= sourceLeft || sourceBottom <= sourceTop) {
            return { visible: 0, loaded: 0, level };
        }

        const tileLeft = Math.max(0, Math.floor((sourceLeft * levelScale) / tileSize));
        const tileTop = Math.max(0, Math.floor((sourceTop * levelScale) / tileSize));
        const tileRight = Math.min(level.columns - 1, Math.floor((sourceRight * levelScale) / tileSize));
        const tileBottom = Math.min(level.rows - 1, Math.floor((sourceBottom * levelScale) / tileSize));

        let visible = 0;
        let loaded = 0;
        for (let ty = tileTop; ty <= tileBottom; ty++) {
            for (let tx = tileLeft; tx <= tileRight; tx++) {
                visible += 1;
                const tile = getTile(manifest, level, tx, ty);
                const levelX = tx * tileSize;
                const levelY = ty * tileSize;
                const sourceX = levelX / levelScale;
                const sourceY = levelY / levelScale;
                const drawX = (sourceX * frame.scale + frame.padX) * state.zoom + drawOffsetX;
                const drawY = (sourceY * frame.scale + frame.padY) * state.zoom + drawOffsetY;

                if (tile.loaded) {
                    loaded += 1;
                    const drawW = (tile.image.naturalWidth / levelScale) * frame.scale * state.zoom;
                    const drawH = (tile.image.naturalHeight / levelScale) * frame.scale * state.zoom;
                    ctx.drawImage(tile.image, drawX, drawY, drawW, drawH);
                } else if (!tile.failed) {
                    const fallbackW = Math.min(tileSize, level.width - levelX) / levelScale * frame.scale * state.zoom;
                    const fallbackH = Math.min(tileSize, level.height - levelY) / levelScale * frame.scale * state.zoom;
                    ctx.fillStyle = "#181818";
                    ctx.fillRect(drawX, drawY, fallbackW, fallbackH);
                }
            }
        }
        return { visible, loaded, level };
    }

    function strokeCompareFrame(drawOffsetX, drawOffsetY, manifest) {
        const frame = getFrame(manifest);
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 1;
        ctx.strokeRect(drawOffsetX, drawOffsetY, baseWidth * state.zoom, baseHeight * state.zoom);

        if (frame.padX > 0.01 || frame.padY > 0.01) {
            ctx.strokeStyle = "rgba(255,255,255,0.28)";
            ctx.strokeRect(
                drawOffsetX + frame.padX * state.zoom,
                drawOffsetY + frame.padY * state.zoom,
                frame.width * state.zoom,
                frame.height * state.zoom,
            );
        }
    }

    function draw() {
        state.raf = 0;
        const dpr = window.devicePixelRatio || 1;
        const size = getCanvasSize();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, size.width, size.height);
        ctx.fillStyle = "#101010";
        ctx.fillRect(0, 0, size.width, size.height);

        let statsA;
        let statsB;
        if (state.mode === "side_by_side") {
            const half = size.width / 2;
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, 0, half, size.height);
            ctx.clip();
            statsA = drawTiles(manifestA, state.offsetX, state.offsetY, 0, 0, half, size.height);
            strokeCompareFrame(state.offsetX, state.offsetY, manifestA);
            ctx.restore();

            ctx.save();
            ctx.beginPath();
            ctx.rect(half, 0, half, size.height);
            ctx.clip();
            statsB = drawTiles(manifestB, half + state.offsetX, state.offsetY, half, 0, half, size.height);
            strokeCompareFrame(half + state.offsetX, state.offsetY, manifestB);
            ctx.restore();

            ctx.strokeStyle = "rgba(255,255,255,0.35)";
            ctx.beginPath();
            ctx.moveTo(half, 0);
            ctx.lineTo(half, size.height);
            ctx.stroke();
        } else if (state.mode === "difference") {
            statsA = drawTiles(manifestA, state.offsetX, state.offsetY, 0, 0, size.width, size.height);
            ctx.save();
            ctx.globalCompositeOperation = "difference";
            statsB = drawTiles(manifestB, state.offsetX, state.offsetY, 0, 0, size.width, size.height);
            ctx.restore();
            strokeCompareFrame(state.offsetX, state.offsetY, manifestA);
        } else {
            statsA = drawTiles(manifestA, state.offsetX, state.offsetY, 0, 0, size.width, size.height);
            const splitX = Math.round(size.width * state.split);
            ctx.save();
            ctx.beginPath();
            ctx.rect(splitX, 0, size.width - splitX, size.height);
            ctx.clip();
            statsB = drawTiles(manifestB, state.offsetX, state.offsetY, splitX, 0, size.width - splitX, size.height);
            ctx.restore();
            ctx.strokeStyle = "rgba(255,255,255,0.72)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(splitX, 0);
            ctx.lineTo(splitX, size.height);
            ctx.stroke();
            strokeCompareFrame(state.offsetX, state.offsetY, manifestA);
        }

        const visible = (statsA?.visible || 0) + (statsB?.visible || 0);
        const loaded = (statsA?.loaded || 0) + (statsB?.loaded || 0);
        const levelA = statsA?.level?.level ?? "-";
        const levelB = statsB?.level?.level ?? "-";
        status.textContent = `${state.mode} | ${Math.round(state.zoom * 100)}% | levels A:${levelA} B:${levelB} | tiles ${loaded}/${visible}`;
    }

    function scheduleDraw() {
        if (!state.raf) {
            state.raf = requestAnimationFrame(draw);
        }
    }

    function setMode(mode) {
        state.mode = mode;
        splitInput.style.display = mode === "slider" ? "" : "none";
        fit();
    }

    splitInput.oninput = () => {
        state.split = Number(splitInput.value) / 100;
        scheduleDraw();
    };

    buttons.appendChild(makeButton("Fit", fit));
    buttons.appendChild(makeButton("100%", setHundredPercent));
    buttons.appendChild(makeButton("Slider", () => setMode("slider")));
    buttons.appendChild(makeButton("Side", () => setMode("side_by_side")));
    buttons.appendChild(makeButton("Diff", () => setMode("difference")));
    buttons.appendChild(splitInput);
    buttons.appendChild(makeButton("-", () => zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 0.8)));
    buttons.appendChild(makeButton("+", () => zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 1.25)));
    buttons.appendChild(makeButton("Close", closeViewer));
    header.appendChild(title);
    header.appendChild(buttons);

    canvasWrap.addEventListener("mousedown", (event) => {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvasWrap.style.cursor = "grabbing";
    });
    onMouseMove = (event) => {
        if (!state.dragging) {
            return;
        }
        state.offsetX += event.clientX - state.lastX;
        state.offsetY += event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        scheduleDraw();
    };
    onMouseUp = () => {
        state.dragging = false;
        canvasWrap.style.cursor = "grab";
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    canvasWrap.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        zoomAt(x, y, event.deltaY < 0 ? 1.18 : 0.85);
    }, { passive: false });

    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            closeViewer();
        }
    });
    closeOnEscape = function (event) {
        if (event.key === "Escape" && document.body.contains(overlay)) {
            closeViewer();
        }
    };
    document.addEventListener("keydown", closeOnEscape);
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    setMode(state.mode);
}

app.registerExtension({
    name: "NH.Nodes.LargeImagePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === PREVIEW_NODE_NAME) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                ensurePreviewButton(this);
                return result;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                ensurePreviewButton(this);
                return result;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                this.__nhLargePreviewManifest = message?.nh_large_preview || [];
            };
            return;
        }

        if (nodeData.name === COMPARE_NODE_NAME) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                ensureCompareButton(this);
                return result;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                ensureCompareButton(this);
                return result;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                this.__nhLargeCompareManifest = message?.nh_large_compare || [];
            };
        }
    },
});
