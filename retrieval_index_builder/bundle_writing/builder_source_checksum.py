# Checksum over the builder's own source files, stored in the manifest (the project has no git history to point at)
import hashlib  # checksum
from pathlib import Path  # source folders

BUILDER_FOLDER = Path(__file__).resolve().parents[1]  # retrieval_index_builder/
ENCODER_FOLDER = BUILDER_FOLDER.parent / "gpt2_service_huggingface_space" / "retrieval_encoders"  # the encoders that make the vectors
EXCLUDED_FOLDER_NAMES = frozenset({"tests", "build_output", "__pycache__"})  # generated or test-only folders do not change the bundle


def compute_builder_source_sha256() -> str:  # one checksum over file names and bytes of every builder and encoder .py file, in sorted order
    digest = hashlib.sha256()  # running checksum
    for source_folder in (BUILDER_FOLDER, ENCODER_FOLDER):  # builder first, then the encoders
        for source_path in sorted(source_folder.rglob("*.py")):  # every module in a stable order
            if EXCLUDED_FOLDER_NAMES & set(source_path.relative_to(source_folder).parts):  # a test, a generated file or a cache
                continue  # not part of the build logic
            digest.update(source_path.relative_to(source_folder).as_posix().encode("utf-8"))  # the name, so a rename changes the checksum
            digest.update(source_path.read_bytes())  # the code
    return digest.hexdigest()  # hex string
