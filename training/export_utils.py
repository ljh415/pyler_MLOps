import os
import re
from glob import glob

import onnx

ONNX_TO_TRITON_DTYPE = {
    1: "TYPE_FP32",    # ONNX FLOAT -> Triton TYPE_FP32
    2: "TYPE_UINT8",
    3: "TYPE_INT8",
    4: "TYPE_UINT16",
    5: "TYPE_INT16",
    6: "TYPE_INT32",
    7: "TYPE_INT64",
    9: "TYPE_BOOL",
    10: "TYPE_FP16",   # ONNX FLOAT16 -> Triton TYPE_FP16
    11: "TYPE_FP32",   # ONNX FLOAT32 -> Triton TYPE_FP32
    12: "TYPE_FP64",   # ONNX FLOAT64 -> Triton TYPE_FP64
    13: "TYPE_UINT32",
    14: "TYPE_UINT64",
    15: "TYPE_COMPLEX64",
    16: "TYPE_COMPLEX128",
}

def get_model_info(model_path):
    match = re.match(r"model_(\d{6}-\d{4})_(\d+\.\d+)\.pth", os.path.basename(model_path))
    if match:
        timestamp, acc = match.groups()
        return timestamp, float(acc)
    return None, None

def find_best_model(model_dir):
    model_files = glob(os.path.join(model_dir, "model_*.pth"))

    if not model_files:
        return None

    valid_models = [m for m in model_files if get_model_info(m)[1] is not None]
    if not valid_models:
        return None
    
    best_model = max(model_files, key=lambda x: get_model_info(x)[1])
    return best_model

def update_best_model_symlink(model_dir, new_model, log_messages):
    best_model_symlink = os.path.join(model_dir, "best_model.pth")

    new_timestamp, new_acc = get_model_info(new_model)

    if os.path.exists(best_model_symlink) or os.path.islink(best_model_symlink):
        existing_model = os.path.realpath(best_model_symlink)
        existing_timestamp, existing_acc = get_model_info(existing_model)

        if existing_acc > new_acc or (existing_acc == new_acc and existing_timestamp > new_timestamp):
            return None
        

        if os.path.islink(best_model_symlink) or os.path.exists(best_model_symlink):
            os.remove(best_model_symlink)
        
        os.symlink(new_model, best_model_symlink)
        log_messages.append(f"Updated best model symlink → {new_model}")
        return new_model
    else:
        os.symlink(new_model, best_model_symlink)
        log_messages.append(f"Created best model symlink → {new_model}")
        return new_model
    
def get_next_model_version(model_base_path):
    existing_versions = [int(d) for d in os.listdir(model_base_path) if d.isdigit()]
    
    if not existing_versions:
        return "1"
    
    return str(max(existing_versions) + 1)

def generate_triton_config(
    model_path: str,
    model_name: str = "resnet18_mnist",
    max_batch_size: int = 16,
    instance_count: int = 1,
    preferred_batch_sizes: list = [2, 4, 8]
):

    model = onnx.load(model_path)
    graph = model.graph

    # ✅ 배치 차원을 제외한 입력 및 출력 노드 정보 추출
    inputs = []
    for inp in graph.input:
        dtype = inp.type.tensor_type.elem_type  # ONNX 데이터 타입
        triton_dtype = ONNX_TO_TRITON_DTYPE.get(dtype, "TYPE_FP32")  # Triton 데이터 타입 변환

        dims = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]

        # ✅ 배치 차원 제거 (첫 번째 차원)
        if len(dims) > 1:
            dims = dims[1:]

        inputs.append(f"""
    {{
        name: "{inp.name}"
        data_type: {triton_dtype}
        dims: {dims}
    }}""")

    outputs = []
    for out in graph.output:
        dtype = out.type.tensor_type.elem_type  # ONNX 데이터 타입
        triton_dtype = ONNX_TO_TRITON_DTYPE.get(dtype, "TYPE_FP32")  # Triton 데이터 타입 변환

        dims = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]

        # ✅ 배치 차원 제거 (첫 번째 차원)
        if len(dims) > 1:
            dims = dims[1:]

        outputs.append(f"""
    {{
        name: "{out.name}"
        data_type: {triton_dtype}
        dims: {dims}
    }}""")

    config_pbtxt = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [{",".join(inputs)}
]

output [{",".join(outputs)}
]

instance_group [
    {{
        count: {instance_count}
        kind: KIND_CPU
    }}
]

dynamic_batching {{
    preferred_batch_size: {preferred_batch_sizes}
}}

optimization {{
    input_pinned_memory {{ enable: true }}
    output_pinned_memory {{ enable: true }}
}}
"""

    return config_pbtxt