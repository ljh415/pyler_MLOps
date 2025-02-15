import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

import sys
sys.stdout.reconfigure(line_buffering=True)

TRITON_GRPC_URL = "triton-service:8001"

MODEL_NAME = "resnet18_mnist"
SAMPLING_NUM = 10

def _init_connection():
    client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
    
    if not client.is_server_live():
        raise RuntimeError("Triton Server is not live!")
    if not client.is_server_ready():
        raise RuntimeError("Triton Server is not ready!")
    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model {MODEL_NAME} is not ready!")
    
    print(f"Triton Server is live and model {MODEL_NAME} is ready")
    return client

def get_model_io_shapes(client:grpcclient.InferenceServerClient, model_name: str):
    model_config = client.get_model_config(model_name, as_json=True)
    model_config = model_config["config"]

    def extract_io_info(io_data):
        return {
            "name": io_data["name"],
            "shape": [SAMPLING_NUM] + list(map(int, io_data["dims"])),
            "dtype": io_data["data_type"].replace("TYPE_", "")
        }

    return {
        "input": extract_io_info(model_config["input"][0]),
        "output": extract_io_info(model_config["output"][0])
    }

def get_image(input_shape:list, input_dtype:str):
    np_dtype = np.float32 if input_dtype == "FP32" else np.float16
    return np.random.rand(*input_shape).astype(np_dtype)

def inference():
    try:
        client = _init_connection()

        io_dict = get_model_io_shapes(client, MODEL_NAME)
        
        input_data = get_image(io_dict["input"]["shape"], io_dict["input"]["dtype"])
        input_tensor = grpcclient.InferInput(io_dict["input"]["name"], io_dict["input"]["shape"], io_dict["input"]["dtype"])
        input_tensor.set_data_from_numpy(input_data)
        
        output_tensor = grpcclient.InferRequestedOutput(io_dict["output"]["name"])
        
        start_time = time.time()
        results = client.infer(model_name=MODEL_NAME, inputs=[input_tensor], outputs=[output_tensor])
        latency = (time.time() - start_time) * 1000
        
        output_data = results.as_numpy(io_dict["output"]["name"])
        predicted_classes = np.argmax(output_data, axis=1)
        
        if len(predicted_classes) < SAMPLING_NUM:
            print(f"Warning: Received {len(predicted_classes)} results, expected {io_dict['input']['shape'][0]}")
        
        # 결과 출력
        print(f"Inference Latency: {latency:.2f} ms", flush=True)
        print(f"Inference Results (10 Samples): {predicted_classes}")
        
    except InferenceServerException as e:
        print(f"Failed to perform inference: {str(e)}")
    except Exception as e:
        print(f"General Error: {str(e)}")

if __name__ == "__main__":
    inference()