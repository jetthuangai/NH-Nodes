# Modified by NH-Nodes: sessions come from ..ort_session (CPU fallback + install hint).
import os
import onnxruntime as ort
from ..ort_session import create_session
from .parsing_api import onnx_inference


class Parsing:
    def __init__(self, model_root, device):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = create_session(os.path.join(model_root, 'humanparsing/parsing_atr.onnx'), device,
                                      sess_options=session_options)
        self.lip_session = create_session(os.path.join(model_root, 'humanparsing/parsing_lip.onnx'), device,
                                          sess_options=session_options)

    def __call__(self, input_image):
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
