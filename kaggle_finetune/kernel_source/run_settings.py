# Every switch of the Kaggle fine-tune in one place; the run mode picks smoke (sanity ladder), full (real run) or dry (local CPU)
import os  # reads the run mode the bundle sets
from dataclasses import dataclass, field, replace  # settings record and per-mode copies

RUN_MODE = os.environ.get("GOT_RUN_MODE", "smoke")  # smoke, full, dry or dry_full; written by build_kernel_bundle.py
FIXED_QUESTIONS = (  # the same 10 questions for the current and the new model
    "Who is the mother of dragons?",
    "What are the words of House Stark?",
    "Who is Tyrion Lannister?",
    "What is the Night's Watch?",
    "Who killed King Joffrey?",
    "What is the name of Arya Stark's sword?",
    "Who is Hodor?",
    "What happened at the Red Wedding?",
    "Who is Brienne of Tarth?",
    "What did Cersei Lannister do to keep power in King's Landing?",
)


@dataclass(frozen=True)
class RunSettings:  # all knobs of one run
    run_mode: str = RUN_MODE  # which mode these settings belong to
    model_size: dict = field(default_factory=lambda: {"embedding_dim": 1024, "layer_count": 24, "head_count": 16})  # GPT-2 medium
    dropout_rate: float = 0.05  # same dropout as Karam's Drive notebook
    load_current_checkpoint: bool = True  # continue from the current fine-tuned weights (dry mode uses a tiny random model)
    question_answer_batch_size: int = 32  # short examples per batch
    book_batch_size: int = 4  # 1024-token book windows per batch
    book_window_tokens: int = 1024  # model context length
    max_question_answer_tokens: int = 256  # longer pairs are cut (very rare)
    book_window_fraction: float = 0.5  # half the book windows per epoch (fresh half each time): about 20% of steps, data review item 5
    mask_prompt_tokens: bool = True  # loss only on the answer and its end token
    peak_learning_rate: float = 3e-5  # gentle: the model is already fine-tuned
    warmup_fraction: float = 0.05  # first 5% of steps ramp the learning rate up
    minimum_learning_rate_ratio: float = 0.1  # cosine decay ends at 10% of the peak
    weight_decay: float = 0.1  # AdamW pull toward zero (not on biases and norms)
    gradient_clip_norm: float = 1.0  # stops the loss spikes of the old runs
    use_fp16: bool = True  # mixed precision on the T4
    gradient_checkpointing: bool = True  # recompute block activations in backward: about 30% slower, fits 4 x 1024 tokens on a T4
    max_epochs: int = 5  # ceiling; early stopping usually ends sooner
    evaluation_every_steps: int = 300  # validation pass every N steps
    early_stopping_patience: int = 5  # stop after this many evaluations without improvement
    minimum_improvement: float = 0.002  # smaller drops in validation loss do not count
    max_training_hours: float = 4.0  # hard stop for training, leaves time for the final comparison
    drop_unnamed_subject_pairs: bool = True  # data review fix (a): "the character / speaker / man" questions
    drop_source_citing_pairs: bool = True  # data review fix (a): questions about "the chunk / passage / text"
    drop_pairs_without_named_entity: bool = True  # data review fix (a): no name or place in the question
    drop_near_duplicate_pairs: bool = True  # data review fix (b): same fact again (word overlap >= 0.6)
    drop_repeated_questions: bool = True  # data review fix (b): same question text again (keeps the first)
    min_answer_words: int = 8  # data review: answers under 8 words are dropped
    cut_interpretive_tails: bool = True  # data review: ", indicating ..." endings are cut off
    decoded_example_count: int = 3  # sanity check 1
    answer_loss_expected_range: tuple = (2.3, 3.7)  # sanity check 2: data review expects answer-only loss about 3.0
    legacy_loss_expected_range: tuple = (1.3, 2.4)  # sanity check 2: Karam's all-token validation loss was about 1.8
    overfit_example_count: int = 16  # sanity check 3: one tiny batch
    overfit_steps: int = 150  # sanity check 3 steps
    overfit_learning_rate: float = 1e-4  # higher rate so one batch is memorised quickly
    overfit_pass_loss: float = 0.1  # sanity check 3 passes below this
    slice_question_answer_count: int = 2000  # sanity check 4: small slice of pairs
    slice_book_window_count: int = 80  # sanity check 4: small slice of book windows
    slice_steps: int = 300  # sanity check 4 steps
    slice_evaluation_every_steps: int = 50  # sanity check 4 validation interval
    slice_validation_pair_count: int = 320  # validation subset during the slice
    slice_validation_window_count: int = 24  # validation book windows during the slice
    max_gradient_norm_sane: float = 100.0  # sanity check 4: larger pre-clip norms mean trouble
    stop_pass_count: int = 8  # sanity check 5: at least 8 of 10 answers must end with the end token
    temperature: float = 0.7  # sampling, same for both models
    top_k: int = 40  # sampling
    top_p: float = 0.9  # sampling
    repetition_penalty: float = 1.1  # sampling
    max_new_tokens: int = 120  # longest answer
    generation_seed: int = 1234  # prompt i uses seed 1234 + i for both models
    random_seed: int = 123  # data order, as in Karam's notebooks


def build_run_settings(run_mode=RUN_MODE):  # settings for one mode
    settings = RunSettings(run_mode=run_mode)  # defaults are the full-run values
    if run_mode.startswith("dry"):  # local CPU crash test with a tiny random model (dry = ladder, dry_full = full-run path)
        settings = replace(  # shrink everything
            settings, model_size={"embedding_dim": 64, "layer_count": 2, "head_count": 2}, load_current_checkpoint=False,
            use_fp16=False, overfit_steps=20, slice_question_answer_count=200, slice_book_window_count=8, slice_steps=24,
            slice_evaluation_every_steps=8, slice_validation_pair_count=64, slice_validation_window_count=4, max_new_tokens=20,
            question_answer_batch_size=8, book_batch_size=2, evaluation_every_steps=10, max_epochs=1, max_training_hours=0.01,
        )
    return settings  # ready to use
