# tests/support — AGENTS.md
Purpose: test doubles shared by the evaluation tests, so no test needs the Space service, a model or a real bundle.

## Files
- `fixture_retrieval_fakes.py` — a fake search-settings record, fake evidence records, a fake retriever that answers from the shared fixture's recorded rankings (hybrid, dense only, lexical only, and a stand-in reranker; it remembers every question it was asked), and a fake answer service that records each call (question, evidence object, layout, budget, generator). Attribute names follow the design's service types, so the runners treat them like the real objects.
- `fixture_stage_inputs.py` — golden sets, a fake runtime record and settings (temporary results folder) built from the six fixture questions, so the stage tests run end to end.
- `test_fixture_rank_fusion.py` — recomputes the fixture's rank-fusion example and every recorded hybrid order with a small reference fusion, and checks that the fake retriever returns exactly that order.
