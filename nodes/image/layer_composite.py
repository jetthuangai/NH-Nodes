"""Layer Layout Composite (NH) and Layer Stack Composite (NH): paste images onto a canvas."""

import re

from PIL import Image

from ...core.image_ops import parse_color, pil_to_image_tensor, resize_pil
from ...core.safe_eval import evaluate
from ...core.utils import tensor_to_pil


_MAX_LAYER_STACK_INPUTS = 64


def _resize_cover(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = max(target_width / max(width, 1), target_height / max(height, 1))
    resized = resize_pil(image, (max(1, round(width * scale)), max(1, round(height * scale))), "lanczos")
    left = max(0, (resized.width - target_width) // 2)
    top = max(0, (resized.height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height))


def _resize_contain(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    return resize_pil(image, (max(1, round(width * scale)), max(1, round(height * scale))), "lanczos")


def _paste_layer(canvas, layer, position):
    """Paste ``layer`` onto ``canvas`` through its own alpha, so transparent areas leave the canvas untouched."""
    if layer.mode != "RGBA":
        canvas.paste(layer, position)  # opaque, whatever the canvas mode
    elif canvas.mode == "RGBA":
        # Both have alpha: a straight paste would blend the alpha channels as if they were colours.
        box = (position[0], position[1], position[0] + layer.width, position[1] + layer.height)
        canvas.paste(Image.alpha_composite(canvas.crop(box), layer), position)
    else:
        canvas.paste(layer, position, layer)


def _fit_layer_image(image, target_size, resize_mode):
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Layer slot width and height must be greater than 0")
    if resize_mode == "stretch":
        return resize_pil(image, (target_width, target_height), "lanczos"), 0, 0
    if resize_mode == "contain":
        fitted = _resize_contain(image, (target_width, target_height))
        return fitted, (target_width - fitted.width) // 2, (target_height - fitted.height) // 2
    fitted = _resize_cover(image, (target_width, target_height))
    return fitted, 0, 0


def _grid_slots(canvas_width, canvas_height, columns, rows, margin_x, margin_y, gap_x, gap_y):
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    content_width = max(1, canvas_width - (2 * margin_x) - (gap_x * (columns - 1)))
    content_height = max(1, canvas_height - (2 * margin_y) - (gap_y * (rows - 1)))
    cell_width = max(1, content_width // columns)
    cell_height = max(1, content_height // rows)
    slots = []
    right_edge = canvas_width - margin_x
    bottom_edge = canvas_height - margin_y

    for row in range(rows):
        for column in range(columns):
            x = margin_x + column * (cell_width + gap_x)
            y = margin_y + row * (cell_height + gap_y)
            width = max(1, (right_edge - x) if column == columns - 1 else cell_width)
            height = max(1, (bottom_edge - y) if row == rows - 1 else cell_height)
            slots.append((x, y, width, height))
    return slots


def _relative_slot(canvas_width, canvas_height, x, y, width, height):
    return (
        int(round(canvas_width * x)),
        int(round(canvas_height * y)),
        int(round(canvas_width * width)),
        int(round(canvas_height * height)),
    )


def _preset_layout_slots(preset, canvas_width, canvas_height, margin_x, margin_y, gap_x, gap_y):
    content_x = margin_x
    content_y = margin_y
    content_width = max(1, canvas_width - (2 * margin_x))
    content_height = max(1, canvas_height - (2 * margin_y))

    if preset == "1_full":
        return [(content_x, content_y, content_width, content_height)]
    if preset == "2_vertical":
        return _grid_slots(canvas_width, canvas_height, 1, 2, margin_x, margin_y, gap_x, gap_y)
    if preset == "2_horizontal":
        return _grid_slots(canvas_width, canvas_height, 2, 1, margin_x, margin_y, gap_x, gap_y)
    if preset == "3_vertical":
        return _grid_slots(canvas_width, canvas_height, 1, 3, margin_x, margin_y, gap_x, gap_y)
    if preset == "4_grid":
        return _grid_slots(canvas_width, canvas_height, 2, 2, margin_x, margin_y, gap_x, gap_y)
    if preset == "2_top_1_bottom":
        top_height = max(1, round((content_height - gap_y) * 0.42))
        bottom_y = content_y + top_height + gap_y
        bottom_height = max(1, content_y + content_height - bottom_y)
        half_width = max(1, (content_width - gap_x) // 2)
        return [
            (content_x, content_y, half_width, top_height),
            (content_x + half_width + gap_x, content_y, max(1, content_width - half_width - gap_x), top_height),
            (content_x, bottom_y, content_width, bottom_height),
        ]
    if preset == "2x3_mixed":
        row_heights = [0.20, 0.20, 0.44]
        usable_height = max(1, content_height - (2 * gap_y))
        heights = [max(1, round(usable_height * ratio / sum(row_heights))) for ratio in row_heights]
        heights[-1] = max(1, content_height - (2 * gap_y) - heights[0] - heights[1])
        half_width = max(1, (content_width - gap_x) // 2)
        slots = []
        y = content_y
        for height in heights:
            slots.append((content_x, y, half_width, height))
            slots.append((content_x + half_width + gap_x, y, max(1, content_width - half_width - gap_x), height))
            y += height + gap_y
        return slots
    return _grid_slots(canvas_width, canvas_height, 2, 3, margin_x, margin_y, gap_x, gap_y)


def _parse_custom_layout(custom_layout, canvas_width, canvas_height, custom_units):
    slots = []
    for raw_line in str(custom_layout or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        values = [float(value) for value in re.split(r"[\s,;]+", line) if value.strip()]
        if len(values) < 4:
            raise ValueError(f"Invalid custom layout line: {raw_line}")
        if len(values) >= 5:
            values = values[-4:]
        x, y, width, height = values[:4]

        units = custom_units
        if units == "auto":
            units = "relative" if max(abs(x), abs(y), abs(width), abs(height)) <= 1.0 else "pixels"
        if units == "relative":
            slot = _relative_slot(canvas_width, canvas_height, x, y, width, height)
        elif units == "percent":
            slot = _relative_slot(canvas_width, canvas_height, x / 100.0, y / 100.0, width / 100.0, height / 100.0)
        else:
            slot = (int(round(x)), int(round(y)), int(round(width)), int(round(height)))

        if slot[2] <= 0 or slot[3] <= 0:
            raise ValueError(f"Custom layout slot size must be greater than 0: {raw_line}")
        slots.append(slot)
    return slots


def _layout_slots(layout_mode, layout_preset, canvas_width, canvas_height, columns, rows, margin_x, margin_y, gap_x, gap_y, custom_layout, custom_units):
    if layout_mode == "custom":
        slots = _parse_custom_layout(custom_layout, canvas_width, canvas_height, custom_units)
        if not slots:
            raise ValueError("Custom layout is empty")
        return slots
    if layout_mode == "grid":
        return _grid_slots(canvas_width, canvas_height, columns, rows, margin_x, margin_y, gap_x, gap_y)
    return _preset_layout_slots(layout_preset, canvas_width, canvas_height, margin_x, margin_y, gap_x, gap_y)


def _eval_position_expr(expression, variables):
    # Variable keys are already casefolded. core.safe_eval works in floating point, so 9**9**9 overflows into an
    # error instead of building a huge integer and freezing the worker.
    try:
        return evaluate(expression, variables, zero_division="error", case_insensitive=True)
    except ValueError as exc:  # core.safe_eval.ExpressionError is a ValueError
        raise ValueError(f"Invalid position expression '{expression}': {exc}") from exc


def _split_position_line(line):
    parts = [part.strip() for part in re.split(r"[,;]+", line, maxsplit=1)]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid position line: {line}")
    return parts[0], parts[1]


def _parse_layer_positions(positions, count, background_width=0, background_height=0, layer_sizes=None):
    variables = {
        "a": int(background_width),
        "b": int(background_height),
        "bg_w": int(background_width),
        "bg_h": int(background_height),
        "width": int(background_width),
        "height": int(background_height),
    }
    for index, (layer_width, layer_height) in enumerate(layer_sizes or [], start=1):
        variables[f"w{index}"] = int(layer_width)
        variables[f"h{index}"] = int(layer_height)

    parsed = []
    for raw_line in str(positions or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        x_expr, y_expr = _split_position_line(line)
        x = int(round(_eval_position_expr(x_expr, variables)))
        y = int(round(_eval_position_expr(y_expr, variables)))
        parsed.append((x, y))

        index = len(parsed)
        variables[f"x{index}"] = x
        variables[f"y{index}"] = y

    while len(parsed) < count:
        parsed.append((0, 0))
    return parsed[:count]


class NH_LayerLayoutComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "canvas_width": ("INT", {"default": 2480, "min": 1, "max": 20000, "step": 1}),
                "canvas_height": ("INT", {"default": 3508, "min": 1, "max": 20000, "step": 1}),
                "background_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "layout_mode": (["preset", "grid", "custom"],),
                "layout_preset": (["2x3", "2x3_mixed", "2_top_1_bottom", "1_full", "2_vertical", "2_horizontal", "3_vertical", "4_grid"],),
                "resize_mode": (["cover", "contain", "stretch"],),
                "columns": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "margin_x": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "margin_y": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "gap_x": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "gap_y": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1}),
                "custom_units": (["auto", "relative", "percent", "pixels"],),
                "custom_layout": ("STRING", {
                    "default": "0.06,0.04,0.39,0.20\n0.55,0.04,0.39,0.20\n0.06,0.28,0.39,0.20\n0.55,0.28,0.39,0.20\n0.06,0.58,0.39,0.36\n0.55,0.58,0.39,0.36",
                    "multiline": True,
                }),
                "background_fit": (["stretch", "cover", "contain"],),
            },
            "optional": {
                "background_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "slot_count", "used_count", "layout_info")
    FUNCTION = "composite"
    CATEGORY = "NH-Nodes/Image"

    @staticmethod
    def _paste_clipped(canvas, layer, x, y):
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(canvas.width, int(x) + layer.width)
        bottom = min(canvas.height, int(y) + layer.height)
        if right <= left or bottom <= top:
            return
        crop_box = (left - int(x), top - int(y), right - int(x), bottom - int(y))
        _paste_layer(canvas, layer.crop(crop_box), (left, top))

    def _create_canvas(self, canvas_width, canvas_height, background_color, background_fit, background_image=None):
        fill = parse_color(background_color, (255, 255, 255))
        # An opaque colour gives the RGB canvas (and RGB output) this node always had; a colour with transparency
        # ('#RRGGBBAA' with AA below FF) gives an RGBA canvas so that the transparency survives.
        canvas = Image.new("RGBA" if fill[3] < 255 else "RGB", (canvas_width, canvas_height), fill)

        if background_image is None:
            return canvas

        background = tensor_to_pil(background_image[0], rgb=True, keep_alpha=True)
        fitted, offset_x, offset_y = _fit_layer_image(background, (canvas_width, canvas_height), background_fit)
        self._paste_clipped(canvas, fitted, offset_x, offset_y)
        return canvas

    def composite(
        self,
        images,
        canvas_width,
        canvas_height,
        background_color,
        layout_mode,
        layout_preset,
        resize_mode,
        columns,
        rows,
        margin_x,
        margin_y,
        gap_x,
        gap_y,
        custom_units,
        custom_layout,
        background_fit,
        background_image=None,
    ):
        canvas_width = int(canvas_width)
        canvas_height = int(canvas_height)
        slots = _layout_slots(
            layout_mode,
            layout_preset,
            canvas_width,
            canvas_height,
            int(columns),
            int(rows),
            int(margin_x),
            int(margin_y),
            int(gap_x),
            int(gap_y),
            custom_layout,
            custom_units,
        )

        canvas = self._create_canvas(canvas_width, canvas_height, background_color, background_fit, background_image)
        used_count = min(int(images.shape[0]), len(slots))
        info_lines = []

        for image_index in range(used_count):
            x, y, width, height = slots[image_index]
            layer = tensor_to_pil(images[image_index], rgb=True, keep_alpha=True)
            fitted, offset_x, offset_y = _fit_layer_image(layer, (int(width), int(height)), resize_mode)
            paste_x = int(x) + offset_x
            paste_y = int(y) + offset_y
            self._paste_clipped(canvas, fitted, paste_x, paste_y)
            info_lines.append(
                f"{image_index}: x={int(x)}, y={int(y)}, w={int(width)}, h={int(height)}, mode={resize_mode}"
            )

        return (pil_to_image_tensor(canvas), len(slots), used_count, "\n".join(info_lines))


class NH_LayerStackComposite:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"layer_{index}": ("IMAGE",)
            for index in range(1, _MAX_LAYER_STACK_INPUTS + 1)
        }
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_count": ("INT", {"default": 1, "min": 0, "max": _MAX_LAYER_STACK_INPUTS, "step": 1}),
                "positions": ("STRING", {
                    "default": "0,0",
                    "multiline": True,
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "used_count", "layer_info")
    FUNCTION = "composite"
    CATEGORY = "NH-Nodes/Image"

    @staticmethod
    def _paste_clipped(canvas, layer, x, y):
        x = int(x)
        y = int(y)
        left = max(0, x)
        top = max(0, y)
        right = min(canvas.width, x + layer.width)
        bottom = min(canvas.height, y + layer.height)
        if right <= left or bottom <= top:
            return False

        crop_box = (left - x, top - y, right - x, bottom - y)
        _paste_layer(canvas, layer.crop(crop_box), (left, top))
        return True

    def composite(self, background_image, layer_count, positions, **kwargs):
        # The output has the channels of the background: RGB stays RGB, an RGBA background gives RGBA.
        background = tensor_to_pil(background_image[0], rgb=True, keep_alpha=True)
        canvas = background.copy()
        requested_count = max(0, min(int(layer_count), _MAX_LAYER_STACK_INPUTS))
        layer_images = {}
        layer_sizes = []
        for index in range(1, requested_count + 1):
            layer = kwargs.get(f"layer_{index}")
            if layer is None:
                layer_sizes.append((0, 0))
                continue
            layer_image = tensor_to_pil(layer[0], rgb=True, keep_alpha=True)
            layer_images[index] = layer_image
            layer_sizes.append((layer_image.width, layer_image.height))

        layer_positions = _parse_layer_positions(
            positions,
            requested_count,
            background.width,
            background.height,
            layer_sizes=layer_sizes,
        )
        used_count = 0
        info_lines = []

        for index in range(1, requested_count + 1):
            layer_image = layer_images.get(index)
            if layer_image is None:
                info_lines.append(f"layer_{index}: missing")
                continue

            x, y = layer_positions[index - 1]
            pasted = self._paste_clipped(canvas, layer_image, x, y)
            if pasted:
                used_count += 1
                info_lines.append(
                    f"layer_{index}: x={x}, y={y}, w={layer_image.width}, h={layer_image.height}"
                )
            else:
                info_lines.append(
                    f"layer_{index}: outside canvas x={x}, y={y}, w={layer_image.width}, h={layer_image.height}"
                )

        return (pil_to_image_tensor(canvas), used_count, "\n".join(info_lines))


NODE_CLASS_MAPPINGS = {
    "NH_LayerLayoutComposite": NH_LayerLayoutComposite,
    "NH_LayerStackComposite": NH_LayerStackComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_LayerLayoutComposite": "Layer Layout Composite (NH)",
    "NH_LayerStackComposite": "Layer Stack Composite (NH)",
}
