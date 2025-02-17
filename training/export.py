import os
import json
import logging

import torch
import torch.nn as nn
from torchvision.models import resnet18

from export_utils import find_best_model, update_best_model_symlink, get_model_info, get_next_model_version

def export(target_type):
    save_path = os.environ.get("SAVE_PATH", "/storage")
    model_dir = os.path.join(save_path, "torch_models")
    model_repo_path = os.path.join(save_path, "models", "resnet18_mnist")
    os.makedirs(model_repo_path, exist_ok=True)

    log_messages = []
    ## best acc model check
    best_model = find_best_model(model_dir)
    if not best_model:
        log_messages.append("No model found in torch_models directory")
        return
    
    ## symlink update
    target_model_path = update_best_model_symlink(model_dir, best_model, log_messages)
    if target_model_path is None:
        log_messages.append("Export process skipped as best model remains unchanged.")
        return
    
    ## get model info
    model_timestamp, model_acc = get_model_info(target_model_path)
    
    log_file = os.path.join(save_path, "logs", f"{model_timestamp}", "export.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    
    for msg in log_messages:
        logger.info(msg)
    
    logger.info(f"Using model: {target_model_path}")
    
    new_version = get_next_model_version(model_repo_path)
    onnx_save_path = os.path.join(model_repo_path, new_version, "model.onnx")
    
    ## model load
    logger.info("Loading model...")
    if target_type=="onnx":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        
        ckpt = torch.load(target_model_path, map_location=device, weights_only=False)
        
        if isinstance(ckpt, dict):
            model = resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(ckpt)
        else :
            model = ckpt
        
        model = model.to(device)
        model.eval()
        
        os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        logger.info(f"ONNX Model saved to {onnx_save_path}")
        
        model_metadata = {
            "export_date": model_timestamp,
            "accuracy": model_acc,
            "source_model": target_model_path
        }
        
        metadata_path = os.path.join(os.path.dirname(onnx_save_path), "metadata.json")
        
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")
        
    else :
        raise Exception("Not supported export type")

if __name__ == "__main__":
    target_type = os.environ.get("EXPORT_TYPE", "onnx")
    export(target_type)