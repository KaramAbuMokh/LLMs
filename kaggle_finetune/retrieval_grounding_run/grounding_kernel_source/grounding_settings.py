# Every switch of the retrieval-grounding run (DESIGN.md section 12), built on the mixed run's settings so the shared modules keep working
import os  # run mode written by the bundle
from dataclasses import dataclass, replace  # settings record and per-mode copies
from mixed_settings import MixedRunSettings  # the mixed run's switches (book blocks, optimizer, fp16, sampling = the Space's settings)

GROUNDING_RUN_MODE = os.environ.get("GOT_GROUNDING_RUN_MODE", "grounding_smoke")  # grounding_smoke, grounding_full or dry_grounding
LIVE_MODEL_SHA256_PREFIX = "528ef774"  # the books + Q&A mixed model that is live in the Space: the only valid start weights


@dataclass(frozen=True)
class GroundingRunSettings(MixedRunSettings):  # the mixed settings plus the grounding knobs (fields with the same name override)
    run_mode: str = GROUNDING_RUN_MODE  # which mode these settings belong to
    grounded_batch_size: int = 4  # grounded sequences per step (about 700 tokens each)
    plain_batches_per_hundred_grounded: int = 21  # plain Q&A batches next to 100 grounded ones: about 15 percent of the stream
    book_batches_per_hundred_grounded: int = 21  # book batches next to 100 grounded ones: about 15 percent of the stream
    book_blocks_per_batch: int = 4  # 512-token book blocks per book batch
    peak_learning_rate: float = 2e-5  # gentler than the 3e-5 of the earlier runs: the model must learn a new input format, not new facts
    max_epochs: int = 1  # one pass over the grounded sequences
    evaluation_every_steps: int = 400  # validation pass every N steps
    early_stopping_patience: int = 3  # stop after this many evaluations without a new best
    max_training_hours: float = 2.5  # hard stop for training
    smoke_steps: int = 60  # steps of the GPU smoke run
    validation_plain_pair_count: int = 320  # answerable held-out Q&A pairs scored for the no-evidence path
    validation_window_count: int = 24  # held-out book windows scored for the voice
    plain_loss_rise_limit: float = 0.15  # the report flags a rise of the plain answer loss above this


def build_grounding_settings(run_mode=GROUNDING_RUN_MODE):  # settings for one mode
    settings = GroundingRunSettings(run_mode=run_mode)  # defaults are the GPU values
    if run_mode.startswith("dry"):  # local CPU crash test with a tiny random model
        settings = replace(  # shrink everything
            settings, model_size={"embedding_dim": 64, "layer_count": 2, "head_count": 2}, load_current_checkpoint=False, use_fp16=False,
            gradient_checkpointing=False, question_answer_batch_size=8, grounded_batch_size=2, book_block_tokens=64, book_overlap_tokens=8,
            smoke_steps=12, evaluation_every_steps=6, max_new_tokens=6, max_epochs=1, max_training_hours=0.01,
            validation_plain_pair_count=16, validation_window_count=2,
        )
    return settings  # ready to use
