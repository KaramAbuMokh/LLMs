# The build command's argument rules: ablation sizes override the defaults, and an ablation bundle can never be uploaded
import pytest  # expected exit
from retrieval_index_builder.build_index_command import build_settings_from_arguments, parse_arguments  # code under test
from retrieval_index_builder.index_build_settings import build_default_settings  # shipped defaults


def test_without_options_the_shipped_defaults_apply():  # the real bundle
    settings = build_settings_from_arguments(parse_arguments([]))  # no options
    assert settings == build_default_settings()  # exactly the defaults
    assert (settings.target_tokens, settings.max_tokens, settings.overlap_sentences) == (200, 230, 1)  # DESIGN.md D2
    assert settings.output_folder.name == "bundle"  # the real bundle folder


def test_ablation_sizes_and_folder_override_the_defaults(tmp_path):  # 128/150 and 256/290 builds
    settings = build_settings_from_arguments(parse_arguments(["--target-tokens", "128", "--max-tokens", "150", "--output-folder", str(tmp_path / "target_128")]))
    assert (settings.target_tokens, settings.max_tokens) == (128, 150)  # sizes from the command line
    assert settings.output_folder == tmp_path / "target_128"  # folder from the command line
    assert settings.overlap_sentences == build_default_settings().overlap_sentences  # everything else unchanged


def test_uploading_an_ablation_bundle_is_refused(tmp_path):  # --upload is only for the default folder
    with pytest.raises(SystemExit):  # must refuse
        build_settings_from_arguments(parse_arguments(["--upload", "--output-folder", str(tmp_path)]))  # code under test
