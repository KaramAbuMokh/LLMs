# Mac script: builds the private Kaggle dataset of the retrieval-grounding run (DESIGN.md section 12): grounded training sequences, a held-out oracle
# validation slice and retrieved validation sequences for the held-out development questions. Run from the project folder:
#   PYTHONPATH=gpt2_service_huggingface_space:. python kaggle_finetune/retrieval_grounding_run/grounding_dataset_builder/build_grounding_dataset.py
import dataclasses  # settings copies
import json  # dataset files
import random  # seeded validation split
import sys  # progress output
from collections import defaultdict  # passages by part
import tiktoken  # GPT-2 tokenizer
from gpt2_demo_service.index_loading.bundle_loading import load_index_bundle  # validated bundle
from gpt2_demo_service.index_search.evidence_retriever import EvidenceRetriever  # the Space's search
from gpt2_demo_service.service_settings import load_service_settings  # shipped search settings
from retrieval_contracts.rare_shingle_location import build_rare_shingle_index  # rare shingles of the seven parts
from retrieval_encoders.pinned_text_encoder import PinnedModelTextEncoder  # the Space's encoder
from retrieval_index_builder.source_alignment.book_part_texts import load_book_part_texts  # the seven part texts
from grounding_data_settings import GroundingDataSettings  # numbers and paths
from grounding_example_building import build_pair_example, build_retrieved_example  # sequences
from oracle_passage_finding import find_oracle_passage, locate_pair_span  # answer passages

MAXIMUM_PASSAGES = 3  # passages the Space packs at most


def build_grounding_dataset(settings: GroundingDataSettings) -> dict:  # writes the dataset files and returns the counts
    tokenizer = tiktoken.get_encoding("gpt2")  # needs the tiktoken cache (TIKTOKEN_CACHE_DIR) or network once
    part_texts = load_book_part_texts(settings.source_data_folder)  # seven parts
    part_lengths = {part_number: len(text) for part_number, text in part_texts.items()}  # sizes for the coarse windows
    shingle_index = build_rare_shingle_index(part_texts)  # about 1 GB, 6 seconds
    encoder = PinnedModelTextEncoder(settings.model_files_folder, thread_count=settings.embedding_thread_count)  # same encoder as the Space
    bundle = load_index_bundle(settings.bundle_folder, encoder.fingerprint)  # the served bundle
    search_settings = dataclasses.replace(load_service_settings({}).evidence_search, pair_examples_enabled=False)  # passages only
    retriever = EvidenceRetriever(bundle, encoder, search_settings, None)  # hybrid search, no reranker
    passages_by_part = defaultdict(list)  # passages grouped by part for the oracle search
    for record in bundle.passages:  # every indexed passage
        passages_by_part[record.part_number].append(record)  # group it
    pairs = json.loads((settings.source_data_folder / "question_answer_train.json").read_text(encoding="utf-8"))  # the 34,001 training pairs
    oracle_examples, distractor_only_examples, skipped_without_oracle = [], [], 0  # outputs and a counter
    for number, pair in enumerate(pairs):  # every training pair
        span = locate_pair_span(pair["q"], pair["a"], pair["book_location"], part_lengths, shingle_index, settings.pair_window_characters)  # where it sits
        oracle = find_oracle_passage(span, passages_by_part, settings.minimum_oracle_hits, settings.held_out_part_number)  # its passage
        if oracle is None:  # no proof of the source passage
            skipped_without_oracle += 1  # count it
            continue  # next pair
        example = build_pair_example(pair, oracle, span, retriever.retrieve(pair["q"]), settings, tokenizer)  # the grounded sequence
        if example is not None:  # packing and retrieval worked
            (oracle_examples if example["has_oracle"] else distractor_only_examples).append(example)  # the split into training and validation comes later
        if number % 2000 == 0:  # progress
            print("pairs %d examples %d" % (number, len(oracle_examples) + len(distractor_only_examples)), file=sys.stderr, flush=True)  # progress line
    return write_dataset_files(settings, distractor_only_examples, oracle_examples, retriever, tokenizer, skipped_without_oracle)  # split, validation sets, write


def write_dataset_files(settings: GroundingDataSettings, distractor_only: list, oracle_examples: list, retriever, tokenizer, skipped_without_oracle: int) -> dict:  # splits off the validation slice and writes
    shuffler = random.Random(settings.random_seed)  # repeatable split
    shuffler.shuffle(oracle_examples)  # random pairs go to validation
    validation = oracle_examples[:settings.oracle_validation_count]  # held out of training
    training = oracle_examples[settings.oracle_validation_count:] + distractor_only  # everything else trains
    shuffler.shuffle(training)  # mixed order on disk
    golden = json.loads(settings.golden_sets_path.read_text(encoding="utf-8"))  # the question sets
    retrieved_validation = [build_retrieved_example(question["question"], question["gold_answer"], retriever.retrieve(question["question"]), settings, tokenizer, MAXIMUM_PASSAGES)
                            for question in golden["held_out_development"]]  # what retrieval really returns for the development questions
    settings.output_folder.mkdir(parents=True, exist_ok=True)  # the payload folder
    for file_name, content in (("grounded_train.json", training), ("grounded_oracle_validation.json", validation), ("grounded_retrieved_validation.json", retrieved_validation)):
        (settings.output_folder / file_name).write_text(json.dumps(content), encoding="utf-8")  # one file per set
    counts = {"grounded_train": len(training), "train_with_oracle": sum(example["has_oracle"] for example in training), "oracle_validation": len(validation),
              "retrieved_validation": len(retrieved_validation), "pairs_without_oracle": skipped_without_oracle,
              "mean_train_tokens": round(sum(len(example["tokens"]) for example in training) / max(1, len(training)), 1)}  # for the report
    (settings.output_folder / "build_report.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")  # kept with the data
    return counts  # printed by the caller


if __name__ == "__main__":  # run as a script
    print(json.dumps(build_grounding_dataset(GroundingDataSettings()), indent=1))  # the counts
