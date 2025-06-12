unset http_proxy && unset https_proxy
export HF_ENDPOINT="https://hf-mirror.com"

while true; do
    # Auto resumed download
    huggingface-cli download Qwen/Qwen3-235B-A22B && break
    echo "Download failed, retrying in 10 seconds..."
    sleep 10
done
