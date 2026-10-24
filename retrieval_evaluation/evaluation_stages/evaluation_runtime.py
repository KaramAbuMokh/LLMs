# The service objects a real run needs, gathered in one record so every stage takes them as arguments (nothing here imports the service;
# run_evaluation_command.py builds the record from the service's public functions)
from dataclasses import dataclass  # frozen record
from pathlib import Path  # bundle folders
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class EvaluationRuntime:  # retriever factory, answer service and helpers
    shipped_search_settings: Any  # the service's default EvidenceSearchSettings (load_service_settings({}).evidence_search)
    build_retriever: Callable[[str, Any], Any]  # (bundle name, EvidenceSearchSettings) -> EvidenceRetriever (bundles and encoders are cached by the builder)
    bundle_folders: Mapping[str, Path]  # bundles that exist on disk, by name ("shipped" first)
    answer_service: Any  # GroundedAnswerService with the 600 second generator limit
    build_answer_request: Callable[[str], Any]  # question -> AnswerRequest with the run's sampling and stop settings
    make_random_generator: Callable[[int], Any]  # seed -> seeded torch generator
    layouts: Mapping[str, Any]  # layout value -> PromptLayout member
    report_progress: Callable[[str], None]  # one line of progress
