
## 1. Envrionment Setup

### 1.1. Docker

Use `pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0`.

For example:

```bash
docker run -td --gpus all --name <container-name> --network host \
    -v $HOME/Projects:/root/Projects  \
    -v /home:/home \
    -v /nvme:/nvme  \
    -e http_proxy=<http-proxy>  \
    -e https_proxy=<https-proxy>  \
    -e hf_token=<your-hf-token>  \
    -e HF_HOME=/nvme/model_hub \
    --shm-size 10G \
    pjlab/deeplearning:v2.2.3-cuda12.8.0-torch2.7.0
```

### 1.2. Python Packages

**INSIDE THE CONTAINER**

```bash
git clone --depth 1 --branch v0.9.0.1.devel https://github.com/jamesnulliu/vllm /path/to/vllm

cd /path/to/vllm

conda install -y cmake=3.26

pip install regex setuptools-scm setuptools-rust flashinfer-python --no-build-isolation

VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -v -e .
```

## 2. Usage

To inference, modify `scripts/inference.sh` and `scripts/inference.py`, and run:

```bash
bash scripts/inference.sh
```

To donwload model/dataset, modify `scripts/download.sh`, and run:

```bash
python scripts/download.sh
```

To show result, modify `scripts/show.py`, and run:

```bash
python scripts/show.py
```