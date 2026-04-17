# Custom Workflow Commands

These are sanitized notes for the custom synthesis pipeline I was using. They are archival reminders, not guaranteed-to-work commands.

Replace placeholders before use:

- `YOUR_TARGET_WEBSITE`
- `YOUR_API_KEY`
- `YOUR_API_BASE`
- any timestamped output path

## Short Workflow

### Step 1

Adjust `--num_seeds` as needed.

```bash
python evolution_synth.py \
  --target_start_url "YOUR_TARGET_WEBSITE" \
  --num_seeds 3 \
  --openai_api_key "YOUR_API_KEY" \
  --openai_api_base "YOUR_API_BASE"
```

### Step 2

Adjust `tasks_path`, `num_processes`, `max_steps`, and `failed_retry` as needed.

```bash
python multi_exeagent.py \
  --model gpt-4o \
  --num_processes 4 \
  --max_steps 15 \
  --failed_retry 1 \
  --tasks_path "outputs/synthagent/custom_evolution/TIMESTAMP/tasks.jsonl" \
  --target_env custom \
  --target_start_url "YOUR_TARGET_WEBSITE" \
  --allow_external_urls True \
  --refine yes \
  --env_start_port 10000 \
  --openai_api_key "YOUR_API_KEY" \
  --openai_api_base "YOUR_API_BASE"
```

### Step 3

```bash
python scoreagent.py \
  --input "outputs/exeagent/webarena/TIMESTAMPED_OUTPUT_FOLDER" \
  --openai_api_key "YOUR_API_KEY" \
  --openai_api_base "YOUR_API_BASE"
```

## Longer Pipeline Reminder

This was another rough version of the same workflow:

1. Run `synthagent.py` for the custom target
2. Run `convert_custom.py`
3. Run `multi_exeagent.py` with refinement enabled
4. Run `scoreagent.py` on the collected output

I removed original local secrets from these notes during cleanup.
