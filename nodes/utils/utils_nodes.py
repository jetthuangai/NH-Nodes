
class NH_UniversalSliderBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("STRING", {
                    "multiline": True,
                    "default": (
                        "# Syntax: name, type, default, min, max, step\n"
                        "# Example:\n"
                        "strength, FLOAT, 0.0, -1.0, 1.0, 0.01\n"
                        "steps, INT, 0, -100, 100, 1"
                    )
                }),
            }
        }

    # This node creates its outputs automatically
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "build_sliders"
    CATEGORY = "NH-Nodes/Utils"
    OUTPUT_NODE = True # Marks this as a node with dynamic outputs
    # This node has no real inputs/outputs, so it cannot be used. DEPRECATED hides it from search,
    # while saved old workflows still load. Use Multi-Slider (FLOAT/INT) instead.
    DEPRECATED = True

    def build_sliders(self, config, **kwargs):
        cls = self.__class__
        cls.RETURN_TYPES = ()
        cls.RETURN_NAMES = ()
        
        # Parse the config to determine the widgets and outputs
        lines = [line.strip() for line in config.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        output_values = []
        # The UI needs its own dictionary to avoid errors
        ui_widgets = {}

        for i, line in enumerate(lines):
            try:
                parts = [p.strip() for p in line.split(',')]
                name, type_str, default, min_val, max_val, step = parts
                
                widget_name = f"value_{i}_{name}"
                type_str = type_str.upper()

                if type_str == "FLOAT":
                    ui_widgets[widget_name] = ("FLOAT", {"default": float(default), "min": float(min_val), "max": float(max_val), "step": float(step), "display": "slider"})
                elif type_str == "INT":
                    display_type = "slider" if (int(max_val) - int(min_val)) < 4096 else "number"
                    ui_widgets[widget_name] = ("INT", {"default": int(default), "min": int(min_val), "max": int(max_val), "step": int(step), "display": display_type})
                
                cls.RETURN_TYPES += (type_str,)
                cls.RETURN_NAMES += (name,)
                
                # Take the actual value from kwargs when present, otherwise use the default
                cast_fn = float if type_str == "FLOAT" else int
                output_values.append(kwargs.get(widget_name, cast_fn(default)))
                
            except Exception as e:
                print(f"[NH-Nodes] Error parsing slider config line: '{line}'. Error: {e}")

        return {"ui": ui_widgets, "result": tuple(output_values)}

class NH_BooleanSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "boolean_switch": ("BOOLEAN", {"default": True, "label_on": "On (True)", "label_off": "Off (False)"}), } }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "get_value"
    CATEGORY = "NH-Nodes/Utils"
    def get_value(self, boolean_switch):
        return (boolean_switch,)

# --- Node registration ---
NODE_CLASS_MAPPINGS = {
    "NH_UniversalSliderBuilder": NH_UniversalSliderBuilder,
    "NH_BooleanSwitch": NH_BooleanSwitch
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NH_UniversalSliderBuilder": "Universal Slider Builder (NH)",
    "NH_BooleanSwitch": "Boolean Switch (NH)"
}