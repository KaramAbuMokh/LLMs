# Upload safety: a public repo is refused, a private one gets one commit, and the token never shows up in results or errors
from types import SimpleNamespace  # simple stand-ins for Hub answers
from unittest import mock  # fake Hub client
import pytest  # expected errors
from retrieval_contracts.shared_fixture.fixture_files import FIXTURE_BUNDLE_FOLDER  # a valid bundle to upload
from retrieval_index_builder.bundle_writing import bundle_upload  # module under test

SECRET_TOKEN = "hf_test_secret_value"  # a fake token to look for in outputs


def build_fake_hub(is_private: bool) -> mock.MagicMock:  # a HfApi class whose instances answer like a repo that is (not) private
    fake_api = mock.MagicMock()  # the client instance
    fake_api.repo_info.return_value = SimpleNamespace(private=is_private)  # the repo's visibility
    fake_api.upload_folder.return_value = SimpleNamespace(oid="a" * 40)  # the commit made by the upload
    return mock.MagicMock(return_value=fake_api)  # the class: calling it returns the instance


def test_public_repo_is_refused_and_nothing_is_uploaded():  # the books are copyrighted
    fake_class = build_fake_hub(is_private=False)  # a public repo
    with mock.patch.object(bundle_upload, "HfApi", fake_class):  # swap the Hub client
        with pytest.raises(bundle_upload.PublicRepositoryError) as error_info:  # must refuse
            bundle_upload.upload_index_bundle(FIXTURE_BUNDLE_FOLDER, "someone/index", SECRET_TOKEN)  # code under test
    fake_class.return_value.upload_folder.assert_not_called()  # no commit was made
    assert SECRET_TOKEN not in str(error_info.value)  # the token is not in the message


def test_private_repo_gets_one_commit_and_a_receipt():  # the normal path
    fake_class = build_fake_hub(is_private=True)  # a private repo
    with mock.patch.object(bundle_upload, "HfApi", fake_class):  # swap the Hub client
        receipt = bundle_upload.upload_index_bundle(FIXTURE_BUNDLE_FOLDER, "someone/index", SECRET_TOKEN)  # code under test
    fake_api = fake_class.return_value  # the client instance
    assert fake_api.create_repo.call_args.kwargs["private"] is True  # a new repo is created private
    assert fake_api.upload_folder.call_count == 1  # one commit
    assert sorted(fake_api.upload_folder.call_args.kwargs["allow_patterns"]) == sorted(["manifest.json", *bundle_upload.BUNDLE_DATA_FILE_NAMES])  # only bundle files
    assert (receipt.repo_id, receipt.commit_hash, receipt.index_version) == ("someone/index", "a" * 40, "got-retrieval-20261024-eeb6e60a")  # receipt content
    assert SECRET_TOKEN not in repr(receipt)  # the receipt never carries the token


def test_damaged_bundle_is_refused_before_any_hub_call(tmp_path):  # a bundle that fails its own check never leaves the Mac
    for source_file in FIXTURE_BUNDLE_FOLDER.iterdir():  # copy the fixture bundle
        if source_file.is_file() and source_file.suffix in {".json", ".jsonl", ".npy", ".npz"}:  # data files only
            (tmp_path / source_file.name).write_bytes(source_file.read_bytes())  # copy
    (tmp_path / "chapters.jsonl").write_text("changed\n", encoding="utf-8")  # damage one file
    fake_class = build_fake_hub(is_private=True)  # a private repo
    with mock.patch.object(bundle_upload, "HfApi", fake_class):  # swap the Hub client
        with pytest.raises(ValueError):  # must refuse
            bundle_upload.upload_index_bundle(tmp_path, "someone/index", SECRET_TOKEN)  # code under test
    fake_class.assert_not_called()  # the Hub was never contacted
