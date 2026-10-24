# Fetches the private retrieval bundle from the Hugging Face dataset repo at one pinned commit
import re  # commit hash pattern
from pathlib import Path  # local folder of the downloaded files
from huggingface_hub import snapshot_download  # downloads a repo snapshot into the local cache
from retrieval_contracts.bundle_manifest import BUNDLE_DATA_FILE_NAMES, MANIFEST_FILE_NAME  # the eight file names

FULL_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")  # a full commit hash, never a branch name


def download_index_bundle(repo_id: str, revision: str, token: str | None) -> Path:  # local folder holding the manifest and the seven data files
    if not FULL_COMMIT_PATTERN.fullmatch(revision):  # an unpinned or abbreviated revision could change under us
        raise ValueError("index revision must be a full 40-character commit hash")  # refuse before any network call
    downloaded_folder = snapshot_download(  # only the bundle's own files, nothing else the repo may hold
        repo_id=repo_id, repo_type="dataset", revision=revision, token=token,
        allow_patterns=[MANIFEST_FILE_NAME, *BUNDLE_DATA_FILE_NAMES],
    )
    return Path(downloaded_folder)  # the snapshot folder
