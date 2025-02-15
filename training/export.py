import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

def export(target_type):
    
    save_path = os.environ.get("SAVE_PATH", "/storage")
    model_save_path = os.path.join(save_path, "model.pth")
    onnx_save_path = os.path.join(save_path, "models", "resnet18_mnist", "1", "model.onnx")  # onnx로 변환은 export.py에서 진행할 예정

    print("Model init..", flush=True)
    if target_type=="onnx":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        
        ckpt = torch.load(model_save_path, map_location=device, weights_only=False)
        
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
        print(f"ONNX Model saved to {onnx_save_path}", flush=True)
    else :
        raise Exception("Not supported export type")

if __name__ == "__main__":
    target_type = os.environ.get("EXPORT_TYPE", "onnx")
    export(target_type)