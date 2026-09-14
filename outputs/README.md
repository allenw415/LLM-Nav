# Experiment Outputs

`outputs/` contains local datasets and experiment results. Generated payloads
are ignored by Git; this index and `experiment_manifest_715.json` are the two
tracked metadata files.

For the thesis-facing organization, start at
`thesis_experiments/README.md`. That directory follows Chapter 4 of the thesis
and exposes the canonical files through lightweight symbolic links. The source
files stay at their original paths so existing evaluation scripts and recorded
provenance remain valid.

## Canonical 715-case Dataset

- Test set: `navigation_paths/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl`
- Cases: `TEST001` through `TEST715`, with 715 unique records.

## Required 715-case Results

### Ground-truth Offset and Trajectory Ablation

- Without trajectory: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/geometry_localization_stages_combined715/results.jsonl`
- With trajectory: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/geometry_edge_visit_localization_stages_combined715/results.jsonl`
- Paired report: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/without_vs_with_trajectory_combined715/`

### Estimated Offset, Eight Views

- Without trajectory: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_incremental_10_20_30/milestones/per30/without_trajectory/results.jsonl`
- With trajectory: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_incremental_10_20_30/milestones/per30/with_trajectory/results.jsonl`
- Detailed analysis: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_incremental_10_20_30/analysis/per30_estimated_offset/`

### Estimated Offset, Four Views

- Without trajectory: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_cardinal4_incremental_10_20_30/milestones/per30/without_trajectory/results.jsonl`
- With trajectory: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_cardinal4_incremental_10_20_30/milestones/per30/with_trajectory/results.jsonl`
- Four/eight-view comparison: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_cardinal4_incremental_10_20_30/milestones/per30/view_count_comparison/`

### Localization-stage Ablation

- Visual: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/localization_stage_ablation_8view_trajectory_initial_offset715/visual/results.jsonl`
- Bayesian: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/localization_stage_ablation_8view_trajectory_initial_offset715/bayesian/results.jsonl`
- Spatial: the eight-view estimated-offset, with-trajectory result above.
- Comparison: `navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/localization_stage_ablation_8view_trajectory_initial_offset715/comparison/`

## Retention Rules

- Preserve the dataset, the eight raw result files above, their configuration,
  summaries, paired comparisons, provenance, and referenced source results.
- Preserve pano-viewer exports, `renders/`, `artifacts/pano_viewer/`, and the
  project `venv/`.
- A retry, stage, or partial-milestone JSONL may be removed only after a
  record-by-record equality check against its canonical per30 condition.
- Keep non-identical retry history; a matching test ID alone is not sufficient
  proof that a file is redundant.
