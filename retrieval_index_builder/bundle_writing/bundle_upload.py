# Uploads a finished bundle to the PRIVATE Hugging Face dataset repo and refuses to touch a public one (D14)
from dataclasses import dataclass  # frozen receipt
from pathlib import Path  # bundle folder
from huggingface_hub import HfApi  # Hub client
from retrieval_contracts.bundle_manifest import BUNDLE_DATA_FILE_NAMES, MANIFEST_FILE_NAME, find_bundle_problems, manifest_from_json  # the bundle's own checks


class PublicRepositoryError(RuntimeError):  # raised when the target repo is not private
    pass  # no extra data: the type name is the message


@dataclass(frozen=True)  # immutable record
class UploadReceipt:  # what the service needs to pin the index
    repo_id: str  # dataset repo that holds the bundle
    commit_hash: str  # full commit hash: the index revision the service downloads
    index_version: str  # version string inside the manifest


def upload_index_bundle(bundle_folder: Path, repo_id: str, token: str) -> UploadReceipt:  # create the repo if needed, check it is private, upload
    manifest = manifest_from_json((bundle_folder / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))  # the bundle's manifest
    bundle_problems = find_bundle_problems(bundle_folder, manifest)  # never upload a bundle that fails its own check
    if bundle_problems:  # damaged or incomplete bundle
        raise ValueError("bundle does not match its manifest: %s" % bundle_problems)  # reasons only
    hub_api = HfApi(token=token)  # the token is used for the calls and never printed or stored
    hub_api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)  # new repos are private; an existing repo keeps its own setting
    if not hub_api.repo_info(repo_id, repo_type="dataset").private:  # an existing repo may be public
        raise PublicRepositoryError("%s is public; the book index must stay private" % repo_id)  # the books are copyrighted: refuse
    commit_info = hub_api.upload_folder(  # one commit with the manifest and the seven data files
        folder_path=str(bundle_folder), repo_id=repo_id, repo_type="dataset", allow_patterns=[MANIFEST_FILE_NAME, *BUNDLE_DATA_FILE_NAMES],  # what to send: the folder, the private dataset repo and the eight bundle files only
        commit_message="Retrieval bundle %s" % manifest.index_version,  # readable commit title
    )
    return UploadReceipt(repo_id, commit_info.oid, manifest.index_version)  # the commit hash is the index revision
