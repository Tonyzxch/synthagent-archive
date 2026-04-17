# Retrospective Notes

These notes are adapted from a local draft I wrote while testing the custom synthesis workflow.

## Postmill

For the `postmill` website, the overall effect was poor.

- The homepage at `http://18.216.88.140:9999/` was effectively empty.
- Many subpages also had missing content.
- GPT exploration quality was weak on this site.

Observed failure patterns:

- After entering a sub-area like `forums`, the agent could jump back into another top-level area like `wiki` before finishing the current path, which made the evolved task logic become strange.
- This felt hard to solve cleanly.
- A possible improvement would be: once the agent is already inside a deeper page (`depth > 0`), forbid clicks on the top global navigation bar and only allow interactions in the main content area.
- My intuition at the time was that this would probably require prompt control and still might not be reliable.

Other issues:

- In some runs, earlier exploration produced almost no useful evidence, and the final task seemed to be fabricated mainly from the last visible page.
- In several examples, GPT added requirements that looked invented rather than grounded in the page.
- One example incorrectly generated a task about finding a wiki-specific user guide with detailed requirements that were not actually supported by the observed content.

My conclusion then was that website quality mattered a lot. Richer, more coherent sites were much easier for this pipeline.

## Shopping Site

For the shopping site at `http://18.216.88.140:7770`, the situation was better overall, but there was still a recurring issue:

- The mid-trajectory refinement was usually reasonable.
- The final synthesized complex instruction could still introduce extra hallucinated constraints.
- In several examples, the final sentence added fabricated requirements such as ratings or review conditions.

This suggested that the main weakness was not always the exploration itself, but the final reverse-engineering step becoming too speculative.

At the time I felt the prompt should be made more conservative and less willing to invent unsupported requirements.

## Classifieds

For `CLASSIFIEDS` at `http://18.216.88.140:9980`, the main issue was depth.

- With an exploration depth of `3`, the generated tasks often felt immature.
- A few more steps were usually needed before the synthesized task became specific enough.

Aside from that, the overall pattern was similar to the shopping site:

- the intermediate refinement looked acceptable
- the final task generator could still inject extra requirements

## General Thought

One broader concern I wrote down was whether purely click-oriented trajectories would age badly if websites changed later.

If a product page disappears or moves, the data may become less useful for training or replay. A more robust strategy might need search-oriented recovery behavior rather than depending too heavily on a single brittle path.
