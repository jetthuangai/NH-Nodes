"""Phase 4: Prompt Building — NH_PromptTemplate, NH_PromptScheduler"""

import re
import random


def _as_prompt_text(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _remove_placeholder_preserve_lines(text, name):
    placeholder = r"\{" + re.escape(name) + r"\}"
    horizontal_ws = r"[^\S\r\n]*"
    return re.sub(
        horizontal_ws + r",?" + horizontal_ws + placeholder + horizontal_ws + r",?" + horizontal_ws,
        "",
        text,
    )


def _cleanup_prompt_line(line):
    newline = ""
    body = line
    if body.endswith("\r\n"):
        body = body[:-2]
        newline = "\r\n"
    elif body.endswith("\n") or body.endswith("\r"):
        body = body[:-1]
        newline = line[-1]

    previous = None
    while previous != body:
        previous = body
        body = re.sub(r",[ \t]*,", ",", body)

    body = re.sub(r"^[ \t]*,[ \t]*", "", body)
    body = re.sub(r",[ \t]*$", "", body)
    return body + newline


def _cleanup_prompt_text(text):
    return "".join(_cleanup_prompt_line(line) for line in text.splitlines(keepends=True))


class NH_PromptTemplate:
    """Dien bien vao prompt template voi cac placeholder {name}."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "default": "a {color} {garment}, {style}, 8k photo",
                    "multiline": True,
                }),
                "var_names": ("STRING", {
                    "default": "color,garment,style",
                }),
            },
            "optional": {
                "var1": ("STRING", {"default": "", "forceInput": True}),
                "var2": ("STRING", {"default": "", "forceInput": True}),
                "var3": ("STRING", {"default": "", "forceInput": True}),
                "var4": ("STRING", {"default": "", "forceInput": True}),
                "var5": ("STRING", {"default": "", "forceInput": True}),
                "var6": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "missing_vars")
    FUNCTION = "render"
    CATEGORY = "NH-Nodes/Text"

    def render(self, template, var_names, **kwargs):
        names = [n.strip() for n in var_names.split(",") if n.strip()]
        result = template
        missing = []

        for i, name in enumerate(names):
            value = _as_prompt_text(kwargs.get(f"var{i + 1}"))

            placeholder = "{" + name + "}"
            if placeholder in result:
                if value != "":
                    result = result.replace(placeholder, value)
                else:
                    missing.append(name)
                    result = _remove_placeholder_preserve_lines(result, name)

        result = _cleanup_prompt_text(result)

        return (result, ", ".join(missing))


class NH_PromptScheduler:
    """Chon prompt theo step hien tai."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "default": "prompt 1\nprompt 2\nprompt 3",
                    "multiline": True,
                }),
                "current_step": ("INT", {"default": 0, "min": 0}),
                "total_steps": ("INT", {"default": 10, "min": 1}),
                "mode": (["sequential", "pingpong", "random"],),
            },
            "optional": {
                "seed": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("current_prompt", "progress", "step_index")
    FUNCTION = "schedule"
    CATEGORY = "NH-Nodes/Text"

    def schedule(self, prompts, current_step, total_steps, mode, seed=0):
        lines = [line.strip() for line in prompts.split("\n") if line.strip()]
        if not lines:
            return ("", 0.0, 0)

        n = len(lines)
        progress = min(1.0, current_step / max(total_steps, 1))

        if mode == "sequential":
            # Chia deu total_steps cho N prompts
            steps_per = max(1, total_steps // n)
            idx = min(current_step // steps_per, n - 1)

        elif mode == "pingpong":
            if n == 1:
                idx = 0
            else:
                cycle_len = (n - 1) * 2
                pos = current_step % cycle_len
                if pos < n:
                    idx = pos
                else:
                    idx = cycle_len - pos

        elif mode == "random":
            rng = random.Random(seed + current_step if seed > 0 else None)
            idx = rng.randint(0, n - 1)

        else:
            idx = 0

        idx = max(0, min(idx, n - 1))
        return (lines[idx], progress, idx)


# --- Dang ky ---
NODE_CLASS_MAPPINGS = {
    "NH_PromptTemplate": NH_PromptTemplate,
    "NH_PromptScheduler": NH_PromptScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_PromptTemplate": "Prompt Template (NH)",
    "NH_PromptScheduler": "Prompt Scheduler (NH)",
}
