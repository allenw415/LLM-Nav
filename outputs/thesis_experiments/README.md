# Thesis Chapter 4 Experiments

This is the thesis-facing entry point for the experiment files. Its layout
follows Chapter 4 of `Thesis_睦閎.pdf` (September 2026):

```text
4_1_dataset_and_environment/
4_3_main_results/
4_4_ablations/
  trajectory_memory/
  heading_offset/
  view_count/
  localization_stage/
4_5_failure_analysis/
```

The entries below are symbolic links. They do not duplicate the large JSONL
files. The physical source paths are intentionally unchanged because the
evaluation tools and provenance records still refer to them directly.

## 4.1 Dataset and Navigation Environment

- `test_set_715.jsonl`: the 715 navigation tasks used by every thesis result.
- `test_set_715.csv` and `dataset_summary.json`: inspectable dataset table and
  sampling summary for Table 4-1 and Figures 4-3 to 4-4.
- `panorama_graph.json`: the Level 0 execution graph (1,455 panorama nodes and
  5,058 directed edges in the thesis).
- `pano_room_grounding.json`, `room_graph.json`, and
  `room_graph_with_visual_profiles.json`: room grounding and Room-level Map
  Graph assets.
- `pano_viewer/`: the exported British Museum panorama viewer.
- `dataset_figures/`: gallery-area figures retained as experiment artifacts.

## 4.3 Main Results

The thesis main condition is:

```text
Estimated Heading Offset + 8-view + Trajectory Memory
+ Visual Localization + Bayesian Filtering + Spatial Alignment
```

- `main_results_715.jsonl`: canonical raw output for Table 4-3. It contains all
  715 test IDs; two rows record API errors, so the thesis reports 512 successes
  over the fixed denominator of 715 (SR 71.6%, SPL 0.547).
- `localization_report/`: source for Table 4-4 and Figures 4-5 to 4-6.
- `route_difficulty_grid/`: Visited Room Count x Detour Ratio results for
  Tables 4-5 to 4-6 and Figure 4-8.
- `shortest_path_distance/`: Table 4-7 and Figure 4-7.
- `mean_room_area/`: Table 4-8 and Figure 4-9. This report uses the 713
  completed main-experiment cases when comparing actual agent paths.

## 4.4 Ablations

All linked raw result files contain the same 715 test IDs.

### Trajectory Memory (Tables 4-9 to 4-10)

Fixed conditions: Estimated Heading Offset and 8-view. Only trajectory memory
changes. `with_trajectory_results_715.jsonl` is also the main result.

### Heading Offset (Tables 4-11 to 4-12)

Fixed conditions: 8-view and Trajectory Memory. `estimated_offset` is the main
condition; `ground_truth_offset` uses +30 degrees. The localization report also
contains the exact/incorrect estimated-offset breakdown.

### View Count (Table 4-13)

Fixed conditions: Estimated Heading Offset and Trajectory Memory. Only 4-view
versus 8-view changes. The 8-view result is also the main result.

### Localization Stage (Table 4-14)

Fixed conditions: Estimated Heading Offset, 8-view, and Trajectory Memory.
`visual`, `bayesian`, and `spatial` select the Current Room after
progressively later localization stages. The spatial result is also the main
result.

## 4.5 Failure Analysis

The thesis failure analysis uses Ground-Truth Heading Offset (+30 degrees) with
Trajectory Memory to remove estimated-offset errors from the analysis.

`room_transition_analysis/` contains both gallery-room and stair-node rows. The
workspace summary has 2,527 attempts and 2,128 correct transitions. Table 4-15
excludes `East stairs` (77 correct) and `West stairs` (53 correct), yielding the
thesis totals of 2,397 attempts, 1,998 correct, and 399 wrong (83.4% correct).

`route_divergence_analysis/` and `pano_viewer_review/` are supporting
diagnostic artifacts. They are kept separate from the raw canonical result.

## Canonical Data Rules

- `../experiment_manifest_715.json` stores the SHA-256 digest, size, and record
  count of the dataset and eight distinct canonical raw result files.
- Do not treat `retries/`, `stages/per10`, `stages/per20`, or intermediate
  extension reports as thesis results.
- Do not delete historical source runs merely because their test IDs overlap.
  Some 715-case outputs were assembled from a 485-case source plus a 230-case
  extension, and their provenance still points to those sources.
- The same main result is intentionally linked from multiple ablation folders;
  these are aliases, not duplicated data.
