import os
import re
from glob import glob

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
