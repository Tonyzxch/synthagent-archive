# SynthAgent_new

This repository is an archival fork of [Browser-Syn](https://github.com/Richar-Du/Browser-Syn). I used it for a research-side exploration around instruction data synthesis for web agents, with most of my changes concentrated in [`evolution_synth.py`](./evolution_synth.py) and [`prompts.py`](./prompts.py).

It is not an actively maintained project anymore. I am publishing it mainly as a record of that attempt and as a reference for anyone interested in prompt-driven task synthesis pipelines.

## Lineage

The code history behind this repository is roughly:

- original project: [aiming-lab/SynthAgent](https://github.com/aiming-lab/SynthAgent)
- an adapted branch/repository maintained by my senior: [Richar-Du/Browser-Syn](https://github.com/Richar-Du/Browser-Syn)
- this repository: my own follow-up modifications on top of that intermediate version

So this repo should be understood as a downstream experimental fork rather than an original implementation.

## What This Fork Changed

Compared with the upstream project, this fork mainly experiments with the task-synthesis stage:

- stronger prompt guidance for action selection across exploration, filtering, and verification phases
- stricter task reconstruction prompts that try to infer user intent as explicit constraints instead of replaying surface actions
- an additional `tips` field in synthesized tasks, appended to the final task description as lightweight procedural guidance
- a more lenient trajectory audit prompt focused on semantic alignment instead of literal completion signals

In practice, the most meaningful edits are in:

- [`evolution_synth.py`](./evolution_synth.py)
- [`prompts.py`](./prompts.py)

## Focus of My Work

Most of my actual work in this fork was concentrated in the synthesis logic rather than the rest of the pipeline. If I had to summarize the core idea of my contribution, it would be this:

- make exploration-time action selection more structured
- make synthesized tasks more constraint-driven and less like literal click-trace paraphrases
- make the final task description carry a small amount of procedural guidance
- make quality filtering judge semantic success instead of requiring overly rigid completion signals

More concretely:

- In [`prompts.py`](./prompts.py), I redesigned the prompts around a staged view of browsing behavior: exploration, filtering/navigation, and final verification.
- The action-selection prompt explicitly encourages filter/sort/sub-category operations in the middle phase, which was important for generating more targeted trajectories instead of shallow browsing traces.
- The reverse-engineering prompt shifts from describing surface actions to inferring user intent as constraints. In other words, it tries to convert interaction mechanisms into task semantics.
- I also added a `tips` field so the synthesized instruction can preserve lightweight operational hints about how the task can be completed through the page UI.
- The audit prompt was made intentionally more lenient: if the agent reached the semantically correct target page or relevant final state, that can still count as usable data even without an overly literal final action.

On the implementation side, [`evolution_synth.py`](./evolution_synth.py) is where those prompt-level ideas are turned into actual synthesized examples:

- it threads the staged prompts through exploration
- it uses reverse-engineering at the end of a trajectory to construct the final task
- it appends the generated `tips` back into the stored task text
- it runs an audit pass before deciding whether the sample is high-quality enough to keep

So while this repository contains the full pipeline scaffold, the part that best represents my own research attempt is really the combination of [`prompts.py`](./prompts.py) and the task-finalization path in [`evolution_synth.py`](./evolution_synth.py).

## Repository Status

This repository should be read as:

- a personal research artifact
- a partially cleaned experimental codebase
- a snapshot of an abandoned but meaningful direction

It should not be read as:

- a polished benchmark release
- a fully reproducible end-to-end system without environment-specific setup work
- a maintained package with compatibility guarantees

## Project Structure

- [`synthagent.py`](./synthagent.py): main task synthesis entry
- [`evolution_synth.py`](./evolution_synth.py): exploration and task generation logic
- [`prompts.py`](./prompts.py): prompt definitions for synthesis, refinement, and auditing
- [`multi_exeagent.py`](./multi_exeagent.py): trajectory collection and task refinement
- [`scoreagent.py`](./scoreagent.py): post-hoc trajectory quality scoring
- [`convert_tasks.py`](./convert_tasks.py): merge synthesized tasks
- [`convert_data.py`](./convert_data.py): convert trajectories into training format

## Environment

The codebase was developed around the WebArena-style environment and OpenAI-based prompting. To reproduce anything meaningful, you will likely still need:

- a working WebArena deployment
- Python 3.10
- Playwright
- OpenAI API access

The package metadata is in [`pyproject.toml`](./pyproject.toml), but environment recreation may still require some manual dependency alignment.

## Minimal Workflow

1. Synthesize tasks:

```bash
python synthagent.py \
  --target_env shopping \
  --env_start_port 10000 \
  --synth_until_tasks 500 \
  --openai_api_key "YOUR_API_KEY"
```

If you want the more direct command that I actually remember using during this line of experimentation, it looked like this:

```bash
python evolution_synth.py \
  --target_start_url "YOUR_TARGET_WEBSITE" \
  --num_seeds 100 \
  --openai_api_key "YOUR_API_KEY" \
  --openai_api_base "YOUR_API_BASE"
```

2. Merge task files:

```bash
python convert_tasks.py \
  --start_folder outputs/synthagent \
  --output configs/synthagent.jsonl
```

3. Collect trajectories with task refinement:

```bash
python multi_exeagent.py \
  --num_processes 8 \
  --tasks_path configs/synthagent.jsonl \
  --model gpt-4.1 \
  --ignore_start_url yes \
  --env_start_port 11000 \
  --refine yes \
  --openai_api_key "YOUR_API_KEY"
```

4. Score and refine trajectories:

```bash
python scoreagent.py \
  --input outputs/exeagent/webarena/synthagent.xxxx 
  --openai_api_key "YOUR_API_KEY"
```

5. Convert data for downstream fine-tuning:

```bash
python convert_data.py \
  --input outputs/exeagent/webarena/synthagent.xxxx 
  --output /path/to/output.json
```

## Notes for Readers

- Some files and prompts reflect fast research iteration rather than final software design.
- The checked-in data and outputs are kept mostly for context, not because they represent a canonical release.
- If you are only interested in the synthesis idea, start by reading [`prompts.py`](./prompts.py) and the task-finalization logic in [`evolution_synth.py`](./evolution_synth.py).
- Additional retrospective notes, sample artifacts, and custom-site experiment materials are archived under [`research_archive/`](./research_archive).

## Acknowledgment

This repository builds on the original [SynthAgent](https://github.com/aiming-lab/SynthAgent) line of work and on the later adapted [Browser-Syn](https://github.com/Richar-Du/Browser-Syn) codebase. Credit for the original framework belongs to the upstream authors, and this repository only preserves a later personal experimental branch.

## Personal Note

I am leaving this repository online as a small marker of a past research attempt. It is incomplete, but it mattered to me.
