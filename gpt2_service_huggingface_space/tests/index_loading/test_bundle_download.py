# Tests of the bundle download call: only a full commit is accepted and only the bundle's own files are requested
import pytest  # expected errors and monkeypatching
from gpt2_demo_service.index_loading import bundle_download  # module under test
from retrieval_contracts.bundle_manifest import BUNDLE_DATA_FILE_NAMES, MANIFEST_FILE_NAME  # the eight file names

FULL_COMMIT = "a" * 40  # a well-formed commit hash


def test_download_asks_for_the_dataset_snapshot_at_the_pinned_commit(monkeypatch, tmp_path):  # arguments of the Hub call
    received = {}  # what the fake saw
    monkeypatch.setattr(bundle_download, "snapshot_download", lambda **arguments: received.update(arguments) or str(tmp_path))  # fake Hub
    folder = bundle_download.download_index_bundle("owner/private-index", FULL_COMMIT, "secret-token")  # call
    assert folder == tmp_path  # the snapshot folder comes back
    assert received["repo_type"] == "dataset" and received["revision"] == FULL_COMMIT and received["repo_id"] == "owner/private-index"  # pinned dataset
    assert sorted(received["allow_patterns"]) == sorted([MANIFEST_FILE_NAME, *BUNDLE_DATA_FILE_NAMES])  # only the bundle files
    assert received["token"] == "secret-token"  # the token reaches the Hub call


@pytest.mark.parametrize("revision", ["", "main", "5c38ec7c", "G" * 40, "a" * 39])  # branch names and short or malformed hashes
def test_unpinned_revision_is_refused_before_any_network_call(monkeypatch, revision):  # no network call happens
    monkeypatch.setattr(bundle_download, "snapshot_download", lambda **arguments: pytest.fail("network call"))  # must not be reached
    with pytest.raises(ValueError, match="40-character"):  # clear reason
        bundle_download.download_index_bundle("owner/private-index", revision, None)  # call
