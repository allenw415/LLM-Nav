# Localization Evaluation Report

日期：2026-05-21

本報告模板用於記錄 `localization` evaluation 的設定、命令、輸出與結果摘要。此檔目前不包含尚未實測的 benchmark 數字。

## 評測目標

`eval_localization.py` 用來評估 pano perception 產生的 `visual_localization` 結果，並將預測 room 與既有 room label 進行比較。

主要流程：
- 從 `pano_room_grounding.json` 或既有輸出資料夾抽樣 pano。
- 必要時呼叫 `st_nav.cli.run_pano_perception` 產生 per-pano JSON。
- 讀取 `visual_localization.predicted_room_id`、room score distribution 與 optional spatial alignment 結果。
- 輸出 top-k accuracy、per-room accuracy、missing output count 等 summary。

## 執行命令

基本執行：

```bash
python3 tools/evaluation/localization/eval_localization.py --samples-per-room 5 --seed 0
```

重用既有 per-pano output：

```bash
python3 tools/evaluation/localization/eval_localization.py --reuse-existing-output --samples-per-room 5 --seed 0
```

啟用 spatial alignment：

```bash
python3 tools/evaluation/localization/eval_localization.py --enable-spatial-alignment --samples-per-room 5 --seed 0
```

## 重要輸入

預設 normalized artifacts：

```text
dataset/sites/british_museum/normalized
```

預設 grounding 對照：

```text
dataset/sites/british_museum/normalized/pano_room_grounding.json
```

## 預設輸出

Per-pano output：

```text
outputs/pano_perception_grounding_eval/
```

Render output：

```text
renders/pano_perception_grounding_eval/
```

Summary output：

```text
outputs/pano_perception_grounding_eval/summary.json
```

## 待補結果

下一次完整執行 evaluation 後，建議補上：

- 測試日期與模型設定
- `samples_per_room`、`seed`、是否 `reuse_existing_output`
- 是否啟用 `spatial_alignment`
- overall accuracy
- top-1 / top-3 / top-5 accuracy
- per-room accuracy
- 主要 failure cases
- 是否有 missing output 或 API error

## 相關檔案

- Evaluation script：[eval_localization.py](/home/allenw4/LLM-Nav/tools/evaluation/localization/eval_localization.py)
- Evaluation README：[README.md](/home/allenw4/LLM-Nav/tools/evaluation/README.md)
