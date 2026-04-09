"""Phase 4: Prompt Building — NH_PromptTemplate, NH_PromptScheduler"""

import re
import random


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
            value = kwargs.get(f"var{i + 1}")
            if value is None:
                value = ""
            if isinstance(value, (int, float)):
                value = str(value)
            value = value.strip() if value else ""

            placeholder = "{" + name + "}"
            if placeholder in result:
                if value:
                    result = result.replace(placeholder, value)
                else:
                    missing.append(name)
                    # Xoa placeholder + dau phay/space xung quanh
                    result = re.sub(
                        r',?\s*\{' + re.escape(name) + r'\}\s*,?', '', result
                    )

        # Clean: double space, trailing comma, leading comma
        result = re.sub(r'\s{2,}', ' ', result)
        result = result.strip(', ').strip()

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
