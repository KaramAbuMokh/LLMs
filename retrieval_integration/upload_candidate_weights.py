# Uploads a candidate weights file to the PRIVATE weights repo under a NEW name (never overwrites an existing file, so the live model stays for rollback)
# Usage: python retrieval_integration/upload_candidate_weights.py --weights PATH --name got_model_retrieval_grounded.pth
import argparse  # command line
import os  # the token file location from the environment
import hashlib  # checksum for the record
from pathlib import Path  # paths
from huggingface_hub import HfApi  # Hub client

WEIGHTS_REPO_ID = "karamAM/gpt2-got-and-spam-weights"  # private model repo with the two model files
TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", Path.home() / ".cache" / "huggingface" / "token"))  # a file holding one Hugging Face token: set HF_TOKEN_FILE, or run `huggingface-cli login`


def upload_candidate_weights(weights_path: Path, target_name: str) -> dict:  # uploads and returns the commit and checksum
    api = HfApi(token=TOKEN_FILE.read_text().strip())  # authenticated client
    if not api.repo_info(WEIGHTS_REPO_ID, repo_type="model").private:  # the weights repo must stay private
        raise SystemExit("the weights repo is public: refusing to upload")  # stop
    if target_name in api.list_repo_files(WEIGHTS_REPO_ID, repo_type="model"):  # never replace a file that may be live or a rollback copy
        raise SystemExit("%s already exists in the weights repo: pick a new name" % target_name)  # stop
    digest = hashlib.sha256()  # running hash of the file
    with open(weights_path, "rb") as handle:  # binary read in 16 MB pieces
        for piece in iter(lambda: handle.read(1 << 24), b""):  # until the end
            digest.update(piece)  # add the piece
    commit_info = api.upload_file(path_or_fileobj=str(weights_path), path_in_repo=target_name, repo_id=WEIGHTS_REPO_ID, repo_type="model",
                                  commit_message="Add %s (sha256 %s)" % (target_name, digest.hexdigest()[:8]))  # one new file in one commit
    return {"commit": commit_info.oid, "file": target_name, "sha256": digest.hexdigest()}  # the record


if __name__ == "__main__":  # run as a script
    parser = argparse.ArgumentParser(description="Upload a candidate weights file under a new name")  # parser
    parser.add_argument("--weights", required=True, type=Path, help="local model.pth")  # source
    parser.add_argument("--name", required=True, help="new file name in the weights repo")  # target
    options = parser.parse_args()  # parsed
    print(upload_candidate_weights(options.weights, options.name))  # the result
