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
