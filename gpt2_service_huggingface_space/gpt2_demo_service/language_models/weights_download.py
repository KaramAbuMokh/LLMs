# Fetches the two private weight files from the Hugging Face weights repo (cached after the first download)
from pathlib import Path  # local file paths
from huggingface_hub import hf_hub_download  # downloads one file from a Hub repo


def download_weight_files(repo_id: str, got_file_name: str, spam_file_name: str, hub_token: str | None) -> tuple[Path, Path]:  # local paths of both weight files
    got_weights_path = hf_hub_download(repo_id, got_file_name, token=hub_token)  # fetch the text model (1.7 GB)
    spam_weights_path = hf_hub_download(repo_id, spam_file_name, token=hub_token)  # fetch the classifier (0.5 GB)
    return Path(got_weights_path), Path(spam_weights_path)  # local file paths
