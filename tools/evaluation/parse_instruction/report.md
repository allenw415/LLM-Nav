# `parse_instruction` Evaluation Report

Generated at: `2026-05-22 03:31:23 +0800`

## Config

- Parser: `runtime`
- Active profile: `ollama`
- Model: `gemma4:31b`
- API base: `http://127.0.0.1:11434/v1`
- API kind: `chat_completions`
- Effective num ctx: `8192`
- Artifacts dir: `dataset/sites/british_museum/normalized`

## Summary

- Case count: `10`
- Success count: `10`
- Error count: `0`
- Average latency seconds: `38.906`
- Average input tokens: `1078.8`
- Average output tokens: `173.7`
- Average total tokens: `1252.5`
- Average reasoning tokens: `None`
- Total input tokens: `10788`
- Total output tokens: `1737`
- Total tokens: `12525`
- Total reasoning tokens: `None`

## Task Type Counts

| `task_type` | Count |
|---|---:|
| `artwork_gallery_instruction_following_navigation` | 4 |
| `artwork_goal_navigation` | 1 |
| `gallery_goal_navigation` | 3 |
| `gallery_instruction_following_navigation` | 2 |

## Cases

| # | `instruction` | `task_type` | Source | Goals | Waypoints | Latency | Input Tokens | Output Tokens | Total Tokens | Reasoning Tokens | Error |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `Find the way from Room 4 to Room 23.` | `gallery_goal_navigation` | `Room 4` | `Room 23` | - | 17.533 | 1073 | 85 | 1158 | - | - |
| 2 | `Find the way from Room 8 to Room 23.` | `gallery_goal_navigation` | `Room 8` | `Room 23` | - | 13.953 | 1073 | 83 | 1156 | - | - |
| 3 | `Find the way from the Lamassu to the Townley Venus.` | `artwork_goal_navigation` | `Room 6` | `Room 23` | - | 29.086 | 1074 | 160 | 1234 | - | - |
| 4 | `Find the way from the Nereid Monument to Room 18.` | `gallery_goal_navigation` | `Room 17` | `Room 18` | - | 20.128 | 1075 | 121 | 1196 | - | - |
| 5 | `Find the way from Room 4, passing Room 7 and Room 17, to Room 23.` | `gallery_instruction_following_navigation` | `Room 4` | `Room 23` | `Room 7`, `Room 17` | 23.343 | 1084 | 218 | 1302 | - | - |
| 6 | `Find the way from Room 8, passing Room 9, to Room 23.` | `gallery_instruction_following_navigation` | `Room 8` | `Room 23` | `Room 9` | 23.624 | 1079 | 212 | 1291 | - | - |
| 7 | `Find the way from the Bronze Container for Cosmetic Items, passing the Lamassu and the Nereid Monument, to the Townley Venus.` | `artwork_gallery_instruction_following_navigation` | `Room 12` | `Room 23` | `Room 6`, `Room 17` | 145.491 | 1089 | 267 | 1356 | - | - |
| 8 | `Find the way from the Lamassu, passing Room 8, to Room 23.` | `artwork_gallery_instruction_following_navigation` | `Room 6` | `Room 23` | `Room 8` | 41.374 | 1080 | 156 | 1236 | - | - |
| 9 | `Find the way from Room 6, passing the Lamassu, to the Townley Venus.` | `artwork_gallery_instruction_following_navigation` | `Room 6` | `Room 23` | `Room 6` | 26.024 | 1080 | 216 | 1296 | - | - |
| 10 | `Find the way from the Townley Venus, passing Room 17, to the Lamassu.` | `artwork_gallery_instruction_following_navigation` | `Room 23` | `Room 6` | `Room 17` | 48.508 | 1081 | 219 | 1300 | - | - |

## Raw Result

Full JSON output is available from the matching `--output-path` file when provided.
