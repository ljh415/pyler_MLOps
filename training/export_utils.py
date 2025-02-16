import os
import re
from glob import glob

import onnx

def get_model_info(model_path):
    match = re.match(r"model_(\d{6}-\d{4})_(\d+\.\d+)\.pth", os.path.basename(model_path))
    if match:
        timestamp, acc = match.groups()
        return timestamp, float(acc)
    return None, None

def find_best_model(model_dir):
    """ torch_models 디렉토리에서 가장 높은 정확도를 가진 모델을 찾음 """
    model_files = glob(os.path.join(model_dir, "model_*.pth"))

    if not model_files:
        return None  # 저장된 모델이 없음

    valid_models = [m for m in model_files if get_model_info(m)[1] is not None]
    if not valid_models:
        return None
    
    best_model = max(model_files, key=lambda x: get_model_info(x)[1])
    return best_model

def update_best_model_symlink(model_dir, new_model, log_messages):
    """ 
    심볼릭 링크(`best_model.pth`)를 유지하거나 업데이트하는 로직
    - 기존 심볼릭 링크가 없으면 가장 높은 정확도의 모델을 링크로 생성
    - 기존 심볼릭 링크가 있으면 정확도를 비교하여 업데이트 여부 결정
    """
    best_model_symlink = os.path.join(model_dir, "best_model.pth")

    new_timestamp, new_acc = get_model_info(new_model)

    if os.path.exists(best_model_symlink) or os.path.islink(best_model_symlink):
        # 기존 심볼릭 링크가 있는 경우 → 원본 모델과 비교
        existing_model = os.path.realpath(best_model_symlink)  # 심볼릭 링크가 가리키는 실제 파일
        existing_timestamp, existing_acc = get_model_info(existing_model)

        if existing_acc > new_acc or (existing_acc == new_acc and existing_timestamp > new_timestamp):
            # 기존 모델이 더 정확하거나 최신이면 업데이트하지 않음
            # logger.info(f"Best model remains unchanged: {existing_model}")
            return None  # ✅ 기존 모델 유지 → export 함수에서 바로 종료하도록 설정
        

        # 새로운 모델이 더 정확하거나 최신이면 업데이트
        if os.path.islink(best_model_symlink) or os.path.exists(best_model_symlink):
            os.remove(best_model_symlink)
        
        os.symlink(new_model, best_model_symlink)
        # logger.info(f"Updated best model symlink → {new_model}")
        log_messages.append(f"Updated best model symlink → {new_model}")
        return new_model
    else:
        # 심볼릭 링크가 없으면 새롭게 생성
        os.symlink(new_model, best_model_symlink)
        # logger.info(f"Created best model symlink → {new_model}")
        log_messages.append(f"Created best model symlink → {new_model}")
        return new_model
    
def get_next_model_version(model_base_path):
    """ 기존 모델 버전 확인 후 +1 증가하여 새로운 버전 경로 반환 """
    existing_versions = [int(d) for d in os.listdir(model_base_path) if d.isdigit()]
    
    if not existing_versions:
        return "1"  # 첫 번째 버전은 1

    return str(max(existing_versions) + 1)  # 가장 높은 버전 +1

def add_metadata_to_onnx(onnx_path, metadata):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    model = onnx.load(onnx_path)
    
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto(key=key, value=str(value))
        model.metadata_props.append(entry)
    
    onnx.save(model, onnx_path)