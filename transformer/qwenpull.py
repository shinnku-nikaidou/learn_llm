from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2-0.5B",
    local_dir="./data/qwen2-0.5b",
    local_dir_use_symlinks=False,
    revision="main",
)
