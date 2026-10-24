# Test helpers for the answer package: invented evidence of controllable size
import random  # seeded invented sentences
import pytest  # fixtures
from gpt2_demo_service.index_search.evidence_records import EvidenceRoute, RetrievedEvidence, ScoredPair, ScoredPassage  # evidence records
from gpt2_demo_service.language_models.got_text_generator import SamplingSettings  # sampling record
from retrieval_contracts.bundle_records import PassageRecord, QuestionAnswerRecord  # stored records

INVENTED_WORDS = (  # made-up names and plain words; no book text anywhere in the tests
    "Brannoc", "Vell", "Orrin", "Lumen", "Kettle", "heron", "lantern", "harbor", "tower", "cedar", "salt", "glass", "banner", "forge",
    "beacon", "blade", "orchard", "river", "winter", "council", "carried", "counted", "guarded", "painted", "crossed", "remembered",
)
SAMPLING = SamplingSettings(temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1)  # the API defaults


def build_sentence(random_source: random.Random) -> str:  # one invented sentence of 8 to 13 words
    words = [random_source.choice(INVENTED_WORDS) for _ in range(random_source.randint(8, 13))]  # random words
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."  # capitalised, closed with a full stop


def build_passage_text(random_source: random.Random, sentence_count: int) -> str:  # several invented sentences
    return " ".join(build_sentence(random_source) for _ in range(sentence_count))  # one paragraph


def make_passage(passage_id: str, text: str, part_number: int = 0, character_start: int = 0, chapter: tuple[int | None, str | None] = (1, "Mira")) -> ScoredPassage:  # invented passage with its record
    record = PassageRecord(passage_id, part_number, "The Glass Coast", 1, chapter[0], chapter[1], character_start, character_start + len(text), 0, text)  # stored passage
    return ScoredPassage(record, 1, 0.5, 0.0, None)  # scores do not matter to packing


def make_pair(pair_id: str, question: str, answer: str) -> ScoredPair:  # an invented solved example
    return ScoredPair(QuestionAnswerRecord(pair_id, 1, question, answer, "The Glass Coast", 1, 1, "Mira"), 0.9, 1)  # stored pair


def make_evidence(passages: list[ScoredPassage], pairs: list[ScoredPair], route: EvidenceRoute = EvidenceRoute.PASSAGES_AND_PAIRS) -> RetrievedEvidence:  # invented evidence whose candidates equal the gated lists
    return RetrievedEvidence(route, tuple(passages), tuple(passages), tuple(pairs), tuple(pairs), 0.5, {"encode": 1.0})  # candidates equal the gated lists


@pytest.fixture()
def evidence_factory():  # hands the helper functions to tests
    return {"passage": make_passage, "pair": make_pair, "evidence": make_evidence, "text": build_passage_text, "sentence": build_sentence}  # the helper functions by name
