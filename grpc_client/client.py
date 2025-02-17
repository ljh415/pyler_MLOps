import os
import re
import sys
import time
import json
import asyncio
import logging
import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException

TRITON_GRPC_URL = "triton-service:8001"
MODEL_NAME = "resnet18_mnist"
SAMPLING_NUM = 10
ROOT_SAVE_DIR = "/storage"

def setup_logger(timestamp, log_messages):
    logger = logging.getLogger("InferenceLogger")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        log_file = os.path.join(ROOT_SAVE_DIR, "logs", timestamp, "inference.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # 로그 디렉토리 생성
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

    # log_messages에 저장된 로그를 한 번에 출력
    for msg, level in log_messages:
        if level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)

    log_messages.clear()
    
    return logger

async def async_init_connection(log_messages):
    client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)

    if not await client.is_server_live():
        log_messages.append(("Triton Server is not live!", "error"))
        raise RuntimeError("Triton Server is not live!")
    if not await client.is_server_ready():
        log_messages.append(("Triton Server is not ready!", "error"))
        raise RuntimeError("Triton Server is not ready!")
    if not await client.is_model_ready(MODEL_NAME):
        log_messages.append((f"Model {MODEL_NAME} is not ready!", "error"))
        raise RuntimeError(f"Model {MODEL_NAME} is not ready!")
    
    log_messages.append((f"Triton Server is live and model {MODEL_NAME} is ready", "info"))
    return client

async def get_model_io_date(client:grpcclient.InferenceServerClient, model_name: str):
    def _extract_io_info(io_data):
        return {
            "name": io_data["name"],
            "shape": [SAMPLING_NUM] + list(map(int, io_data["dims"])),
            "dtype": io_data["data_type"].replace("TYPE_", "")
        }
        
    model_config = await client.get_model_config(model_name, as_json=True)
    model_config = model_config["config"]
    io_shape = {
        "input": _extract_io_info(model_config["input"][0]),
        "output": _extract_io_info(model_config["output"][0])
    }
    
    model_metadata = await client.get_model_metadata(model_name, as_json=True)
    model_version = sorted(map(int, model_metadata["versions"]))[-1]
    metadata_path = os.path.join(ROOT_SAVE_DIR, "models", MODEL_NAME, str(model_version), "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return io_shape, metadata["export_date"]

def get_image(input_shape:list, input_dtype:str):
    np_dtype = np.float32 if input_dtype == "FP32" else np.float16
    return np.random.rand(*input_shape).astype(np_dtype)

async def async_inference():
    log_messages = []
    try:
        client = await async_init_connection(log_messages)
        io_dict, timestamp = await get_model_io_date(client, MODEL_NAME)
        
        logger = setup_logger(timestamp, log_messages)
        
        input_data = get_image(io_dict["input"]["shape"], io_dict["input"]["dtype"])
        input_tensor = grpcclient.InferInput(io_dict["input"]["name"], io_dict["input"]["shape"], io_dict["input"]["dtype"])
        input_tensor.set_data_from_numpy(input_data)
        
        output_tensor = grpcclient.InferRequestedOutput(io_dict["output"]["name"])
        
        start_time = time.time()
        results = await client.infer(model_name=MODEL_NAME, inputs=[input_tensor], outputs=[output_tensor])
        latency = (time.time() - start_time) * 1000
        
        output_data = results.as_numpy(io_dict["output"]["name"])
        predicted_classes = np.argmax(output_data, axis=1)
        
        if len(predicted_classes) < SAMPLING_NUM:
            warning_msg = f"Warning: Received {len(predicted_classes)} results, expected {io_dict['input']['shape'][0]}"
            logger.warning(warning_msg)
        
        logger.info(f"Inference Latency: {latency:.2f} ms")
        logger.info(f"Inference Results (10 Samples): {predicted_classes}")
        
    except InferenceServerException as e:
        logger.error(f"Failed to perform inference: {str(e)}")

if __name__ == "__main__":
    asyncio.run(async_inference())