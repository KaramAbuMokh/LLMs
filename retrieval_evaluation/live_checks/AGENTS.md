# live_checks — AGENTS.md
Purpose: checks of the deployed Space (DESIGN.md section 11.7). Run only after a deploy, one caller, at most 20 calls: the Space is public and slow. Standard library only.

## Files
- `live_contract_check.py` — sends the portfolio's old body, a spam body and an over-long question, and reports every break of the contract: the Game of Thrones body must have exactly the completion, retrieval flag and sources keys (never a text key), each source exactly four keys, at most one book snippet, the spam body exactly the completion, the long question a 422 with a detail, and the old health keys must remain. The 503 shape is not provoked.
- `live_latency_probe.py` — times up to 20 calls and reads the Server-Timing header per stage; reports when the header is missing.
