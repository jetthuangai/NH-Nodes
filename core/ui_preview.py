"""Pictures a node shows on itself in the ComfyUI editor (its "ui" images), saved to ComfyUI's temp folder.

A node returns ``{"ui": {"images": save_temp_previews(...)}, "result": (...)}`` and the editor draws the pictures
on the node, so nobody has to wire a Preview Image node to see what it did.
"""

import os
import uuid

from PIL import Image

MAX_PREVIEWS = 16  # pictures per run; a long batch shows its first ones


def save_temp_previews(pictures, prefix):
    """Save uint8 RGB arrays as JPEG files in ComfyUI's temp folder; the "ui" image entries for them.

    Outside ComfyUI (tests, scripts) there is no temp folder to serve them from, and nothing is saved.
    """
    try:
        import folder_paths

        folder = folder_paths.get_temp_directory()
    except Exception:
        return []
    os.makedirs(folder, exist_ok=True)
    entries = []
    run = uuid.uuid4().hex[:12]
    for index, picture in enumerate(pictures[:MAX_PREVIEWS]):
        name = f"{prefix}_{run}_{index}.jpg"
        Image.fromarray(picture).save(os.path.join(folder, name), quality=90)
        entries.append({"filename": name, "subfolder": "", "type": "temp"})
    return entries
