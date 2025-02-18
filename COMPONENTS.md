# Components Details
## 1. Train, Export
```bash
training/
├── Dockerfile
├── export_job.yaml
├── export.py
├── export_utils.py
├── network_monitor.py
├── requirements.txt
├── train_job.yaml
└── train.py
```
### **1-1. Training Flow (`train.py`)**
ResNet-18 모델을 MNIST 데이터셋으로 학습  
Kubernetes **Job**을 사용하여 CPU 환경에서 모델을 훈련, 학습 결과는 PVC에 저장

1. **데이터셋 로드**  
   - MNIST 데이터셋을 PVC 로드
   - 네트워크 사용을 최소화하기 위해 외부 다운로드 없이 PVC에 저장된 데이터를 활용

2. **모델 학습 실행**  
   - ResNet-18 모델을 사용하여 MNIST 데이터셋으로 학습을 진행 
   - 학습 과정에서 손실(loss)과 정확도(accuracy)를 측정하며 로그를 기록
   - test데이터셋으로 최종 acc 산출

3. **체크포인트 및 학습 로그 저장**  
   - 학습된 모델의 **PyTorch Checkpoint**를 PVC에 저장
   - `train.log` 파일을 생성하여 학습 과정과 성능 지표를 기록
4. **Job 완료 후 자동 종료**  
   - Kubernetes Job의 특성상, 모델 학습이 완료되면 자동으로 종료

### **1-2. Export Flow (`export.py`, `export_utils.py`)**
1. **PyTorch Checkpoint 로드**  
    - 학습된 모델을 PVC에서 로드
2. **ONNX 형식으로 변환**  
    - 모델을 ONNX 형식으로 변환하고, 추론을 위한 최적화를 수행
    - `/storage/models/{MODEL_NAME}/{version}/model.onnx`
3. **Triton Inference Server에 적합한 설정 파일 생성**  
    - Triton이 모델을 인식할 수 있도록 `config.pbtxt` 및 메타데이터 파일을 생성
        <details>
            <summary> config.pbtxt </summary>

        ```
        name: "resnet18_mnist"
        platform: "onnxruntime_onnx"
        max_batch_size: 16

        input [
            {
                name: "input"
                data_type: TYPE_FP32
                dims: [1, 28, 28]
            }
        ]

        output [
            {
                name: "output"
                data_type: TYPE_FP32
                dims: [10]
            }
        ]

        instance_group [
            {
                count: 2
                kind: KIND_CPU
            }
        ]

        dynamic_batching {
            preferred_batch_size: [2, 4, 8]
        }

        optimization {
            input_pinned_memory { enable: true }
            output_pinned_memory { enable: true }
        }
        ```

        </details>
4. **Export된 모델 저장**  
   - 변환된 ONNX 모델 및 관련 설정 파일을 **PVC에 저장**하여 Triton Inference Server가 로드
   - metadata 저장
        <details>
            <summary> metadata.json </summary>

        ```json
        model_metadata = {
            "export_date": model_timestamp,
            "accuracy": model_acc,
            "source_model": target_model_path
        }
        ```
        </details>
- 기타사항
    - acc기준 성능 비교 후 기존 모델 대비 성능 향상된 모델에 대해서만 export진행
    - `best_model.pth`로 심볼링링크생성
  
## 2. Model Inference

Triton Inference Server를 통해 Export된 모델을 배포하고, gRPC 기반으로 추론을 수행

### **2-1. Triton Inference Server**
Triton Inference Server는 **Kubernetes Deployment**로 실행되며, PVC에서 Export된 ONNX 모델을 불러와 서빙  
gRPC 및 HTTP 프로토콜을 지원하며, 본 프로젝트에서는 **gRPC**를 사용하여 클라이언트와 통신

#### **Triton Inference Server 동작 과정:**
1. **모델 자동 로드**  
    - `None` 모드(`model-control-mode=none`)를 사용하여 서버가 기동될 때 모든 모델을 자동으로 로드
    - 모델 파일(`onnx_model`, `config.pbtxt`)은 PVC에서 로드

2. **Inference 요청 처리**  
    - gRPC 또는 HTTP를 통해 클라이언트 요청을 수신
    - 입력 데이터를 받아 **ONNX 모델을 사용하여 추론을 수행**
    - 예측 결과를 반환

3. **Kubernetes Service를 통한 접근**  
    - Triton Server는 **Cluster IP 서비스**를 통해 gRPC Client와 연결
    - 클라이언트는 Service Endpoint를 호출하여 Inference 요청을 수행 가능


### **2-2. gRPC Client**

gRPC Client는 Triton Inference Server에 Inference 요청을 보내고, 결과를 처리하는 역할  
독립적인 Kubernetes Pod로 실행되며, gRPC 프로토콜을 통해 요청을 수행

#### **gRPC Client 동작 과정:**
1. Triton Server와 연결
    - Kubernetes Service를 통해 Triton Server에 gRPC 요청을 전송

2. Inference 요청 및 응답 수신
    - 입력 데이터를 준비한 후, Triton Server로 Inference 요청
    - Triton Server로부터 예측 결과를 받아 10개 샘플을 추출하여 저장

3. inference Latency 측정 및 로깅
    - 추론 속도(Inference Latency)를 측정하여 성능을 분석
    - 결과 및 Latency 데이터를 PVC에 로그 파일로 저장

## **3. Helm Packaging & Deployment**

### **3-1. Docker Image Management**
본 프로젝트에서는 모델 학습, 변환(Export), 서빙(Triton), 추론(gRPC Client) 등의 모든 프로세스를 컨테이너화하여 **Docker Hub**에 저장하였습니다.

#### **사용한 Images**:
- `training`, `export`: 학습, Onnx Convert [Docker Hub](https://hub.docker.com/r/ljh415/resnet_training)
- `grpc_client`: Triton Server에 요청을 수행하는 클라이언트 컨테이너 [Docker Hub](https://hub.docker.com/r/ljh415/grpc_client)
- `triton`: Triton Inference Server 컨테이너
(`nvcr.io/nvidia/tritonserver:23.04-py3`)

### **3-2) Helm Packaging**
Helm Chart를 사용하여 Kubernetes 환경에서 모든 프로세스를 자동 배포하도록 패키징하였으며, `initContainer`를 사용하여 각 컴포넌트의 실행 순서를 보장
```
PVC → Training → Export → Triton → gRPC Client
```
