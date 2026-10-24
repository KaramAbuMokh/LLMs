# Every switch of the mixed books + question-answer run, built on the morning run's settings so the shared modules keep working
import os  # run mode written by the bundle
from dataclasses import dataclass, replace  # settings record and per-mode copies
from run_settings import RunSettings  # the morning run's switches (optimizer, fp16, sampling = the Space's settings)

MIXED_RUN_MODE = os.environ.get("GOT_MIXED_RUN_MODE", "mixed_smoke")  # mixed_smoke, mixed_full, dry_smoke or dry_full
MORNING_MODEL_SHA256 = "db0aa51fa2475b42a0a77aa326f8d2c6794dc85bdfbaf35fe9d36509c4b724b5"  # the Space's model since 10-24 11:43
ORIGINAL_MODEL_SHA256 = "cb134705474346ba2944e1444d1c43c7ce4d9a96984cbe89150a7de614365525"  # Karam's original got_model.pth


@dataclass(frozen=True)
class MixedRunSettings(RunSettings):  # the morning settings plus the mixing knobs (fields with the same name override)
    # Start weights: smoke v1 (10-24 12:46) showed the morning model is already at its best on these pairs (every mix raised the
    # answerable held-out loss 2.08 -> 2.12 in 200 steps), so the mixed stream starts from Karam's original, like this morning's run
    start_model: str = "original"  # "original" or "morning"
    book_block_tokens: int = 512  # scored length of one book block
    book_overlap_tokens: int = 64  # each block repeats the last 64 tokens of the one before as unscored context
    book_blocks_per_step: int = 0  # 0 = worked out so every pair and every book block is used once per epoch
    # share of each step's loss from the pairs (the rest from the books): smoke v2 picked 0.3 (held-out answer loss tied at 2.106-2.111
    # for 0.3/0.5/0.7, book loss best at 0.3: 2.658 vs 2.674 vs 2.702); 0.3 is about the natural token ratio of one pass over each
    question_answer_loss_weight: float = 0.3
    sweep_loss_weights: tuple = (0.3, 0.5, 0.7)  # mixes tried on the smoke ladder
    sweep_steps: int = 200  # steps per tried mix
    sweep_pair_count: int = 6400  # pairs in the sweep slice (one pass at 32 per step)
    sweep_block_count: int = 1000  # book blocks in the sweep slice
    sweep_validation_pair_count: int = 320  # answerable held-out pairs scored after each tried mix
    sweep_validation_window_count: int = 24  # held-out book windows scored after each tried mix
    sweep_tie_margin: float = 0.005  # validation losses this close count as a tie; the tie goes to more book text
    overfit_pair_count: int = 8  # tiny-batch check: pairs
    overfit_block_count: int = 2  # tiny-batch check: book blocks
    overfit_steps: int = 120  # tiny-batch check: steps
    overfit_pass_loss: float = 0.3  # both the pair loss and the book loss must end below this
    max_epochs: int = 3  # ceiling; early stopping on the answerable held-out pairs usually ends sooner
    evaluation_every_steps: int = 250  # validation pass every N steps
    early_stopping_patience: int = 4  # stop after this many evaluations without a new best
    max_training_hours: float = 3.0  # hard stop for training, leaves time for scoring three models
    fact_samples_per_question: int = 3  # each fact question is sampled 3 times; 2 right of 3 = right
    expected_morning_answer_loss: float = 2.146  # the morning model's answer loss on the 1,233 held-out pairs (10-24 report)
    expected_morning_book_loss: float = 2.659  # the morning model's loss on the held-out book passage
    starting_loss_tolerance: float = 0.02  # the re-measured numbers must be this close (proves the right weights and data)


def build_mixed_settings(run_mode=MIXED_RUN_MODE):  # settings for one mode
    settings = MixedRunSettings(run_mode=run_mode)  # defaults are the GPU values
    if run_mode.startswith("dry"):  # local CPU crash test with a tiny random model
        settings = replace(  # shrink everything
            settings, model_size={"embedding_dim": 64, "layer_count": 2, "head_count": 2}, load_current_checkpoint=False,
            use_fp16=False, gradient_checkpointing=False, question_answer_batch_size=8, book_block_tokens=64, book_overlap_tokens=8,
            overfit_steps=15, sweep_steps=12, sweep_pair_count=96, sweep_block_count=40, sweep_validation_pair_count=16,
            sweep_validation_window_count=2, max_new_tokens=6, evaluation_every_steps=10, max_epochs=1, max_training_hours=0.005,
        )
    return settings  # ready to use
