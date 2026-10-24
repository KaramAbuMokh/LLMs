# Downloads the pinned public model files (ONNX graph + tokenizer) into one folder; also the image build command
import argparse  # command line for the Docker build step
from pathlib import Path  # target folders
from huggingface_hub import hf_hub_download  # exact-commit file download
from retrieval_contracts.encoder_contracts import PASSAGE_PAIR_SCORER_FILES, TEXT_ENCODER_FILES, PinnedModelFiles  # the two pinned models


def model_folder_for(files: PinnedModelFiles, base_folder: Path) -> Path:  # base/<model id with "/" replaced by "__">
    return base_folder / files.model_id.replace("/", "__")  # one folder per model


def download_model_files(files: PinnedModelFiles, base_folder: Path) -> Path:  # fetches the ONNX graph and tokenizer at the pinned commit
    model_folder = model_folder_for(files, base_folder)  # where this model's files go
    for repo_file_path in (files.model_file, files.tokenizer_file):  # the two files a model needs
        hf_hub_download(repo_id=files.model_id, revision=files.revision, filename=repo_file_path, local_dir=model_folder)  # keeps the repo paths
    return model_folder  # folder holding onnx/model.onnx and tokenizer.json


def main() -> None:  # python -m retrieval_encoders.model_file_download --target <folder>
    argument_parser = argparse.ArgumentParser(description="Download both pinned encoder models.")  # command line definition
    argument_parser.add_argument("--target", type=Path, required=True, help="folder that receives one subfolder per model")  # destination
    arguments = argument_parser.parse_args()  # read the command line
    for files in (TEXT_ENCODER_FILES, PASSAGE_PAIR_SCORER_FILES):  # dense encoder first, then the reranker
        download_model_files(files, arguments.target)  # fetch one model
        print("downloaded %s at %s" % (files.model_id, files.revision[:8]))  # progress line without any path or token


if __name__ == "__main__":  # run as a command
    main()  # start the download
