# Spatial Alignment Ablation: Complex 120-Case Subset

## Comparison and selection

- Without Spatial Alignment: `bayesian_results_715.jsonl` (Current Room selected after Bayesian Filtering).
- With Spatial Alignment: `spatial_results_715.jsonl` (Current Room selected after Spatial Alignment).
- Other ablation settings are fixed, including the estimated initial Heading Offset, 8-view observation, and Trajectory Memory.
- Selected cases: `visited_room_count in {5, 6}` and `detour_ratio in [1.6, 2.0]`.
- The selection contains exactly 120 cases: four sampling cells with 30 cases each.
- All 120 test IDs occur in both result files, and neither condition has an API error in this subset.

## Main result

| Scope | Cases | Without Spatial Alignment | With Spatial Alignment | Change |
|---|---:|---:|---:|---:|
| All 715 cases | 715 | 489/715 (68.4%) | 512/715 (71.6%) | +23, +3.2 pp |
| Complex subset | 120 | 81/120 (67.5%) | 84/120 (70.0%) | +3, +2.5 pp |
| Remaining cases | 595 | 408/595 (68.6%) | 428/595 (71.9%) | +20, +3.4 pp |

The complex 120-case subset does not show a larger aggregate gain than the full test set or the remaining 595 cases.

## Breakdown by Visited Room Count and Detour Ratio

| Visited Room Count | Detour Ratio | n | Without | With | Change |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.6–1.8 | 30 | 14/30 (46.7%) | 14/30 (46.7%) | 0.0 pp |
| 5 | 1.8–2.0 | 30 | 18/30 (60.0%) | 21/30 (70.0%) | +10.0 pp |
| 6 | 1.6–1.8 | 30 | 27/30 (90.0%) | 24/30 (80.0%) | -10.0 pp |
| 6 | 1.8–2.0 | 30 | 22/30 (73.3%) | 25/30 (83.3%) | +10.0 pp |
| **Overall** | **1.6–2.0** | **120** | **81/120 (67.5%)** | **84/120 (70.0%)** | **+2.5 pp** |

### Marginal summaries

| Group | n | Without | With | Change |
|---|---:|---:|---:|---:|
| Visited Room Count = 5 | 60 | 32/60 (53.3%) | 35/60 (58.3%) | +5.0 pp |
| Visited Room Count = 6 | 60 | 49/60 (81.7%) | 49/60 (81.7%) | 0.0 pp |
| Detour Ratio = 1.6–1.8 | 60 | 41/60 (68.3%) | 38/60 (63.3%) | -5.0 pp |
| Detour Ratio = 1.8–2.0 | 60 | 40/60 (66.7%) | 46/60 (76.7%) | +10.0 pp |

## Paired outcome changes

Because the two conditions use the same test IDs, the most informative comparison is paired:

| Without → With | Cases |
|---|---:|
| Failure → Success | 9 |
| Success → Failure | 6 |
| Success → Success | 75 |
| Failure → Failure | 30 |

The net gain is therefore only 3 cases. The paired exact McNemar test for the full 120-case subset gives `p = 0.607`, so this subset does not provide statistically significant evidence of an aggregate improvement.

For the 60 cases with Detour Ratio 1.8–2.0, six cases change from failure to success and none change from success to failure (`+10.0 pp`; exact McNemar `p = 0.031`). This is a positive exploratory signal for highly circuitous routes, but the 1.6–1.8 group changes in the opposite direction and the 30-case cells are small.

## Interpretation

These results do not support the broad claim that Spatial Alignment helps more whenever a case has both more visited rooms and a high detour ratio. The combined complex subset improves by only 2.5 percentage points, slightly less than the overall 3.2-point gain.

A narrower claim is supported: Spatial Alignment appears most helpful in the highest Detour Ratio range (1.8–2.0), where SR rises by 10.0 percentage points for both 5-room and 6-room cases. This should be presented as subgroup evidence rather than a general monotonic complexity trend.

## Discordant test IDs

- Failure → Success: `TEST151`, `TEST190`, `TEST192`, `TEST194`, `TEST242`, `TEST418`, `TEST468`, `TEST476`, `TEST713`
- Success → Failure: `TEST153`, `TEST234`, `TEST238`, `TEST424`, `TEST466`, `TEST705`
