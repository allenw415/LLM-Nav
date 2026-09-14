# Data Tools

Dataset preparation, grounding, and normalization utilities.

These are executable tools, not runtime navigation modules and not persistent
dataset assets.

## Navigation grid sampling

`sample_navigation_grid.py` ports the deterministic room-count by fixed-width
detour-ratio sampler used to build the floor-0 navigation benchmark. It can
preserve an existing partial grid and fill only missing cells. Candidate paths
are shuffled with a fixed seed, then selected greedily to reduce reuse of start
panoramas, start-room/target pairs, room sequences, targets, and start rooms.

Build the 2-to-6-room, 1.0-to-2.0 grid with a per-cell cap of 20 while
preserving TEST001-TEST249:

```bash
venv/bin/python tools/data/sample_navigation_grid.py --allow-sparse-cells
```

The explicit sparse flag is required because `rooms_2/1.6-1.8` and
`rooms_2/1.8-2.0` have only 16 and 9 eligible unique paths, respectively. The
result contains 485 cases: 23 cells contain 20 cases, and the two naturally
undersupplied cells keep their actual denominators. `extension_test_set.jsonl`
contains only TEST250-TEST485 (236 cases), so the existing 249 cases do not need
to be re-evaluated.
