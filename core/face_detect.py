"""Face boxes for Portrait Segment, from YuNet (OpenCV Zoo, MIT license).

YuNet is a 230 KB ONNX model that runs on the CPU through the OpenCV the pack already needs, so finding faces costs
no VRAM and no extra package. The weights are fetched once, pinned to a fixed Hugging Face revision, into the same
``nh_nodes`` model folder as the parsers (models/NH-Nodes/yunet by default).
"""

import os

import cv2
import numpy as np

from .model_files import download_pinned, register_model_folder
from .segformer_backend import FOLDER_EXTENSIONS, FOLDER_NAME, default_folder

YUNET = {
    "repo_id": "opencv/face_detection_yunet",
    "revision": "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
    "file": "face_detection_yunet_2023mar.onnx",
    "folder": "yunet",
}
MIN_SCORE = 0.6  # YuNet's own samples use 0.9; a box that is not a face is turned down later by the face parser
DETECT_SIDES = (2048, 640)  # faces are looked for on copies at most this long: small faces show on the first, big or
# grainy ones (scanned prints, close-ups) are often found only on the second
MAX_FACES = 32  # the largest ones; each face costs two parses

_detector = None


def model_path(download=True):
    """Path of the YuNet weights, from any registered ``nh_nodes`` folder; downloaded into the default one if absent."""
    for base in register_model_folder(FOLDER_NAME, default_folder(), FOLDER_EXTENSIONS):
        candidate = os.path.join(base, YUNET["folder"], YUNET["file"])
        if os.path.isfile(candidate):
            return candidate
    if not download:
        return None
    target = os.path.join(default_folder(), YUNET["folder"])
    download_pinned(YUNET["repo_id"], YUNET["revision"], target, (YUNET["file"], "LICENSE"))
    path = os.path.join(target, YUNET["file"])
    if not os.path.isfile(path):
        raise RuntimeError(f"[NH-Nodes] Face detector missing after download: {path}")
    return path


def _get_detector(size):
    """The YuNet detector, set up for pictures of ``size`` (width, height). Loaded from bytes read by Python, so a
    model folder whose path has accents (which OpenCV cannot open on Windows) works too."""
    global _detector
    if _detector is None:
        weights = np.fromfile(model_path(), dtype=np.uint8)
        _detector = cv2.FaceDetectorYN.create("onnx", weights, np.empty(0, np.uint8), size, MIN_SCORE, 0.3, 5000)
    _detector.setInputSize(size)
    return _detector


def _detect_at(image, max_side, min_score):
    """Face boxes found on a copy of ``image`` at most ``max_side`` pixels long, in the image's pixels."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    small = image
    if scale < 1.0:
        small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    detector = _get_detector((small.shape[1], small.shape[0]))
    _, found = detector.detect(cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_RGB2BGR))
    faces = []
    for row in [] if found is None else found:
        score = float(row[-1])
        if score < min_score:
            continue
        x, y, w, h = (float(v) / scale for v in row[:4])
        box = (max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h))
        if box[2] > box[0] and box[3] > box[1]:
            faces.append((*box, score))
    return faces


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def merge_faces(faces):
    """One box per face from boxes found at several scales: best score first, a box is dropped when it overlaps a
    kept one (IoU 0.3 or more) or lies mostly (60%) inside one, as a half face found at one scale does."""
    kept = []
    for face in sorted(faces, key=lambda f: f[4], reverse=True):
        if all(_overlap(face, other) < 0.3 * (_area(face) + _area(other) - _overlap(face, other))
               and _overlap(face, other) < 0.6 * min(_area(face), _area(other)) for other in kept):
            kept.append(face)
    return kept


def detect_faces(image, min_score=MIN_SCORE, sides=DETECT_SIDES, limit=MAX_FACES):
    """[(x1, y1, x2, y2, score)] of the faces in the uint8 RGB ``image``, in its pixels, largest first.

    Looks on copies of each length in ``sides`` and merges what they find (see merge_faces), which finds faces from
    about 1/150 of the picture's long side up to close-ups. A face cut by the frame is often missed; the caller can
    look again on a copy with a margin, and parses the whole frame when nothing is found.
    """
    found = [face for side in sides for face in _detect_at(image, side, min_score)]
    faces = merge_faces(found)
    faces.sort(key=_area, reverse=True)
    return faces[:limit]
