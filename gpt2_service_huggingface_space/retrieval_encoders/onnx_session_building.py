# Builds the CPU inference session both encoders run on (one place for the thread settings)
from pathlib import Path  # model file path
import onnxruntime  # ONNX graph runner


def build_inference_session(model_path: Path, thread_count: int) -> onnxruntime.InferenceSession:  # one session per loaded model
    session_options = onnxruntime.SessionOptions()  # runtime settings
    session_options.intra_op_num_threads = thread_count  # threads inside one operator (the Space has 2 vCPU)
    session_options.inter_op_num_threads = 1  # operators run one after another
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL  # no parallel branches, so threads never fight torch
    return onnxruntime.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])  # CPU only
