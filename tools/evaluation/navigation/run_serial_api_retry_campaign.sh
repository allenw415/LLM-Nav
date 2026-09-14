#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_ROOT="/home/allenw4/LLM-Nav"
RETRY_TOOL="$PROJECT_ROOT/tools/evaluation/navigation/retry_incremental_api_errors.py"
SOURCE_ROOT="$PROJECT_ROOT/outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_incremental_10_20_30/milestones/per30"
RETRY_PARENT="$PROJECT_ROOT/outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_incremental_10_20_30/retries/milestone_per30_715_api_errors"

CAMPAIGN_ID="${1:?usage: run_serial_api_retry_campaign.sh CAMPAIGN_ID [MAX_ROUNDS] [ROUND_DELAY_SECONDS]}"
MAX_ROUNDS="${2:-5}"
ROUND_DELAY_SECONDS="${3:-1200}"
CAMPAIGN_LOG="$RETRY_PARENT/${CAMPAIGN_ID}.campaign.log"
CAMPAIGN_STATUS="$RETRY_PARENT/${CAMPAIGN_ID}.status.json"

if [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid campaign ID: $CAMPAIGN_ID" >&2
  exit 2
fi
if [[ ! "$MAX_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ROUNDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$ROUND_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "ROUND_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
fi

mkdir -p "$RETRY_PARENT"
cd "$PROJECT_ROOT" || exit 90

write_status() {
  local status="$1"
  local round="$2"
  local run_id="$3"
  local retry_status="$4"
  local without_errors="$5"
  local with_errors="$6"
  local total_errors="$7"
  local status_tmp="${CAMPAIGN_STATUS}.tmp"
  jq -n \
    --arg campaign_id "$CAMPAIGN_ID" \
    --arg status "$status" \
    --arg run_id "$run_id" \
    --argjson round "$round" \
    --argjson max_rounds "$MAX_ROUNDS" \
    --argjson retry_status "$retry_status" \
    --argjson without_errors "$without_errors" \
    --argjson with_errors "$with_errors" \
    --argjson total_errors "$total_errors" \
    --arg updated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
      campaign_id: $campaign_id,
      status: $status,
      round: $round,
      max_rounds: $max_rounds,
      run_id: $run_id,
      retry_status: $retry_status,
      without_errors: $without_errors,
      with_errors: $with_errors,
      total_errors: $total_errors,
      condition_execution: "serial",
      llm_timeout_seconds: 600,
      max_http_retries: 1,
      updated_at: $updated_at
    }' > "$status_tmp"
  mv "$status_tmp" "$CAMPAIGN_STATUS"
}

count_errors() {
  jq -s '[.[] | select(.error != null)] | length' "$1"
}

printf 'campaign_id=%s\ncampaign_pid=%s\nstarted_at=%s\nmax_rounds=%s\nround_delay_seconds=%s\ncondition_execution=serial\nllm_timeout_seconds=600\nmax_http_retries=1\nwithout_key_source=ST_NAV_PROFILE_GEMINI_API_KEY\nwith_key_source=ST_NAV_WITH_TRAJECTORY_API_KEY\nkeys_are_distinct=validated_by_each_round\n' \
  "$CAMPAIGN_ID" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MAX_ROUNDS" "$ROUND_DELAY_SECONDS" > "$CAMPAIGN_LOG"
initial_without_errors="$(count_errors "$SOURCE_ROOT/without_trajectory/results.jsonl")"
initial_with_errors="$(count_errors "$SOURCE_ROOT/with_trajectory/results.jsonl")"
initial_total_errors="$((initial_without_errors + initial_with_errors))"
write_status "starting" 0 "" 0 "$initial_without_errors" "$initial_with_errors" "$initial_total_errors"

for ((round = 1; round <= MAX_ROUNDS; round += 1)); do
  run_id="${CAMPAIGN_ID}_round${round}"
  printf '[%s] round=%s run_id=%s status=starting\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$round" "$run_id" >> "$CAMPAIGN_LOG"

  python3 "$RETRY_TOOL" \
    --stage per30 \
    --scope milestone \
    --run-id "$run_id" \
    --llm-timeout 600 \
    --max-http-retries 1 \
    --serial-conditions >> "$CAMPAIGN_LOG" 2>&1
  retry_status="$?"

  without_errors="$(count_errors "$SOURCE_ROOT/without_trajectory/results.jsonl")"
  with_errors="$(count_errors "$SOURCE_ROOT/with_trajectory/results.jsonl")"
  total_errors="$((without_errors + with_errors))"
  printf '[%s] round=%s retry_status=%s without_errors=%s with_errors=%s total_errors=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$round" "$retry_status" \
    "$without_errors" "$with_errors" "$total_errors" >> "$CAMPAIGN_LOG"

  if ((total_errors == 0)); then
    write_status "complete" "$round" "$run_id" "$retry_status" 0 0 0
    printf '[%s] campaign complete: all API errors resolved\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CAMPAIGN_LOG"
    exit 0
  fi

  write_status "running" "$round" "$run_id" "$retry_status" \
    "$without_errors" "$with_errors" "$total_errors"
  if ((round < MAX_ROUNDS)); then
    printf '[%s] sleeping %s seconds before next round\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ROUND_DELAY_SECONDS" >> "$CAMPAIGN_LOG"
    sleep "$ROUND_DELAY_SECONDS"
  fi
done

without_errors="$(count_errors "$SOURCE_ROOT/without_trajectory/results.jsonl")"
with_errors="$(count_errors "$SOURCE_ROOT/with_trajectory/results.jsonl")"
total_errors="$((without_errors + with_errors))"
write_status "exhausted" "$MAX_ROUNDS" "${CAMPAIGN_ID}_round${MAX_ROUNDS}" 0 \
  "$without_errors" "$with_errors" "$total_errors"
printf '[%s] campaign exhausted with total_errors=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$total_errors" >> "$CAMPAIGN_LOG"
exit 1
