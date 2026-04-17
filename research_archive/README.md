# Research Archive Notes

This folder stores a small set of archival materials from my later experiments around custom website exploration and instruction synthesis.

It is not a polished subproject. I am only keeping the pieces that still seem potentially useful if I ever revisit this direction:

- retrospective notes about failure modes and prompt hallucination
- sanitized command notes for reproducing the rough workflow
- a few synthesized task samples and result files
- a small amount of reference material about candidate websites and intermediate code history

Files that were clearly redundant, externally recoverable, or mostly just temporary runtime output were removed during cleanup.

## Kept Material

- [`notes/retrospective_notes.md`](./notes/retrospective_notes.md): my qualitative observations about failure cases, especially hallucinated constraints in final synthesized tasks
- [`notes/commands.md`](./notes/commands.md): sanitized command snippets for the custom synthesis workflow
- [`notes/candidate_sites.txt`](./notes/candidate_sites.txt): a larger list of possible websites for future experiments
- [`notes/selected_sites.txt`](./notes/selected_sites.txt): a shorter hand-picked subset
- [`samples/`](./samples): a few generated task/result artifacts worth keeping as examples
- [`references/synthagent_browser_syn_snapshot.py`](./references/synthagent_browser_syn_snapshot.py): an older code snapshot kept only as historical reference
- [`references/convert_custom_snapshot.py`](./references/convert_custom_snapshot.py): a one-off helper script from the custom workflow, kept only as an archival reference

## Removed During Cleanup

- local copies of public paper assets that can be recovered elsewhere
- duplicate images already present in the main repository
- a duplicate modified script that matched the current top-level implementation
- verbose run logs that were not as useful as the retained task samples and notes
- generated packaging metadata, scratch image dumps, and unrelated side-experiment files from the repo root
