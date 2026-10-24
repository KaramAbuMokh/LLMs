# monitoring — AGENTS.md
Purpose: numbers about the running service. Standard library plus psutil. The Space log must be treated as public: no visitor text, no book text.

## Files
- `memory_logging.py` — current and peak memory in megabytes for `/health` and the log.
- `stage_timing.py` — the stage stopwatch: wrap a stage in `measure(name)`, read milliseconds per stage (a stage measured twice adds up).
- `request_logging.py` — writes one JSON line per Game of Thrones answer: question length and an 8-hex hash, route, evidence score, passage and pair ids, stage milliseconds, token counts, stop reason, guard flag, index version.
