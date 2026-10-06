# Evaluation Tools

Benchmark and evaluation entrypoints are grouped by evaluation target.

## Parse Instruction

Folder: `tools/evaluation/parse_instruction/`

- `eval_parse_instruction.py`: evaluates the runtime `parse_instruction` parser across instruction cases and reports latency plus input/output/total/reasoning token usage when available.
- `report.md`: stores the latest parse instruction benchmark notes.

Run it with:

```bash
python3 tools/evaluation/parse_instruction/eval_parse_instruction.py --llm-num-ctx 8192
```

Save JSON and generate a Markdown report with:

```bash
python3 tools/evaluation/parse_instruction/eval_parse_instruction.py \
  --output-path outputs/parse_instruction_eval/result.json \
  --report-path outputs/parse_instruction_eval/report.md \
  --llm-num-ctx 8192
```

## Localization

Folder: `tools/evaluation/localization/`

- `eval_localization.py`: evaluates integrated visual localization from pano perception outputs against known room labels.
- `report.md`: stores localization evaluation notes and result summaries.

Run it with:

```bash
python3 tools/evaluation/localization/eval_localization.py --samples-per-room 5 --seed 0 --llm-num-ctx 16384
```

## Floor 0 Navigation

Folder: `tools/evaluation/navigation/`

### Claude Sonnet 5 preflight

Create an Anthropic API key, then store it in the repository `.env` without
passing it on the command line:

```dotenv
ANTHROPIC_API_KEY=your-key-here
```

Validate request conversion locally without making a paid API call:

```bash
python3 tools/evaluation/navigation/preflight_claude_sonnet5.py --mode dry-run
```

After the key is configured, validate native structured output with one normal
Messages request, then validate Message Batches separately:

```bash
python3 tools/evaluation/navigation/preflight_claude_sonnet5.py --mode messages
python3 tools/evaluation/navigation/preflight_claude_sonnet5.py --mode batch
```

The batch preflight stores only secret-free status metadata under
`outputs/navigation_evaluation/claude_sonnet5_preflight/`. It does not print or
write the API key.

Synchronize the benchmark dataset and the existing floor-0 render cache. The command is read-only unless `--apply` is present:

```bash
venv/bin/python tools/evaluation/navigation/sync_floor0_assets.py
venv/bin/python tools/evaluation/navigation/sync_floor0_assets.py --apply
```

Check dataset and render coverage without calling an LLM or Street View image API:

```bash
venv/bin/python tools/evaluation/navigation/evaluate_navigation_batch.py --inventory-only
```

Run a cache-only smoke case. A missing manifest stops the case and does not request an image:

```bash
venv/bin/python tools/evaluation/navigation/evaluate_navigation_batch.py \
  --test-id TEST001 \
  --limit 1 \
  --output-dir outputs/navigation_evaluation/smoke/TEST001
```

Enable lazy rendering only when paid image requests are intended. Each new museum panorama produces eight images and is saved under `renders/room_grounding_fov90` for later cases:

```bash
venv/bin/python tools/evaluation/navigation/evaluate_navigation_batch.py \
  --render-missing
```

The evaluator is sequential and resumable by `test_id`. It writes `config.json`, `results.jsonl`, and `summary.json` below `outputs/navigation_evaluation/`.

Run only the per-20 extension TEST250-TEST485, retry execution errors once,
and build both the 236-case extension report and 485-case combined report:

```bash
venv/bin/python tools/evaluation/navigation/run_extension236.py
```

The runner writes per-case durable progress to `results.jsonl` and updates
`run_status.json` with runner/child PIDs, the current case, the last completed
case, and retry/reporting phases. Re-running the same command resumes from
the first missing test ID.

Analyze the relationship between estimated mean gallery area and success rate
for the paired 715-case Without/With Trajectory experiment:

```bash
venv/bin/python tools/evaluation/navigation/report_navigation_by_mean_room_area.py
```

The report calculates two equal-weight area measures: one over the unique
gallery rooms in `shortest_path_rooms`, and one over the unique ground-truth
gallery rooms traversed by each agent's actual `pano_path`. Circulation nodes
and unlabeled transit panoramas are excluded. Both analyses use only the fixed
`0–200`, `200–400`, `400–600`, and `600–800` m² display intervals. Values above
800 m² are retained in the final display interval and counted explicitly in
the CSV/JSON outputs. The reference-path bins contain the same tests for both
methods; actual-path bin membership is method-specific because the two agents
can traverse different rooms. The output directory contains a 715-row audit
CSV, separate reference/actual fixed-bin CSV and JSON summaries, and two
minimal slide-ready PNG figures. The X axes show the four interval labels, and
the figures draw Wilson 95% confidence intervals. Gallery areas are estimates
rather than architectural floor-plan measurements.

An additional actual-path sensitivity analysis excludes cases above 800 m²
separately for each method before calculating interval SR. It is written as
`actual_agent_path_fixed_200m2_max800m2.{csv,json,png}`; the unfiltered version
is preserved for comparison.

Report room-level first-exit accuracy over all 715 paired cases. Each
condition-test-room contributes at most one first exit attempt. Wrong includes
both entering a non-shortest-next room and never leaving the current room;
the audit CSV retains wrong-room, no-exit, and later-recovery fields:

```bash
venv/bin/python tools/evaluation/navigation/analyze_room_exit_accuracy.py
```

Outputs are written to `room_level_exit_accuracy/` and include the per-attempt
audit table, separate Without/With main tables, long and wide room summaries,
JSON metadata, and a meeting-ready Markdown table.

The same command also reports the estimated total area of the 25 navigable
galleries and writes three distribution figures: reference shortest-path
distance, the sum of unique gallery areas along each reference shortest path,
and the corresponding actual-path area sums for Without/With Trajectory. A
gallery traversed more than once within one path contributes its area only
once. Distance histograms use 20 m bins; area-sum histograms use shared 500 m²
bins so the reference and actual-path distributions remain comparable.

### Map-only passage heading

Evaluate all 76 directed Level 0 adjacent-room edges with only the north-up map,
current room ID, and target room ID. The default repeats every case three times,
for 228 VLM calls:

```bash
venv/bin/python tools/evaluation/navigation/eval_map_grounded_heading.py \
  --repeat 3 \
  --output-dir outputs/map_grounded_heading/all_level0_directed
```

Use the original map without overlays for every case with:

```bash
venv/bin/python tools/evaluation/navigation/eval_map_grounded_heading.py \
  --map-mode plain \
  --repeat 3 \
  --output-dir outputs/map_grounded_heading/plain_map_all_level0_directed
```

Plain mode reuses `level_0_allocentric.png` directly and does not generate
room highlights, C/T markers, a compass sidebar, or per-case map files.

Use `--max-cases 1` for a paid single-case smoke run. Generated reports include
schema validity, exact and ±45-degree accuracy, majority stability, pairwise
agreement, feasibility gates, and a manual-review queue. The evaluator never
sends room-graph headings, panorama data, pano IDs, topology text, or candidate
actions to the VLM.
