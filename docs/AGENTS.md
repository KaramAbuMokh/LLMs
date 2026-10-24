# docs

Purpose: the long-form documents of the retrieval system. Written in Karam's private working project, so a few ids (`wf-1024-2000-got-rag`, `hq.*` keys) and file names in them (the live-system inventory, the Kaggle result folders, the publish tool) point at private notes that are not in this repository.

## Files
- `DESIGN.md`: the design as built: goals and acceptance targets, 24 decisions with their reasons and rejected options, the data flow from the offline build to an answer with sources, the frozen contract, every runtime module with its signatures, the index builder, the evaluation and its decision rules, the file tree, deploy and rollback, and a change log. Section 17 lists what changed after the first release and wins where it differs from the sections above (for example the passage size ended at 128 tokens, not 200).
- `RESULTS.md`: the plain write-up: before and after table, finding the right passage, what is live, live measurements, targets passed and missed, the decisions made on the way, the hardening pass and how to undo each step.
- `AGENTS.md`: this file.
