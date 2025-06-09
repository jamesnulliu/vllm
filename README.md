
## 1. Envrionment Setup

### 1.1. Docker

Use `pjlab/deeplearning:v2.1.1-torch2.7.0-cuda12.6.0-ubuntu24.04`.

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
    --shm-size 20G \
    pjlab/deeplearning:v2.1.0-torch2.7.0-cuda12.6.0-ubuntu24.04
```

### 1.2. Python Packages

**INSIDE THE CONTAINER**

```bash
conda install -y cmake=3.26

git clone --depth 1 --branch v0.9.0.1.devel https://github.com/jamesnulliu/vllm /path/to/vllm
cd /path/to/vllm
pip install regex setuptools-scm setuptools-rust flashinfer-python
pip install --no-build-isolation -v -e .
```