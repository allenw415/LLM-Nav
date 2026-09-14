from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluation.navigation import run_cardinal4_initial_offset_incremental as campaign


TARGET_CONDITION = "with_trajectory"
TARGET_TEST_ID = "TEST004"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_target_error(
    *,
    source_path: Path,
    retry_path: Path,
    target_test_id: str,
) -> dict:
    source_sha256 = _sha256(source_path)
    originals = campaign.load_jsonl(source_path)
    retries = campaign.load_jsonl(retry_path)
    if len(retries) != 1 or retries[0].get("test_id") != target_test_id:
        raise RuntimeError("Retry output must contain exactly the requested test id.")
    replacement = retries[0]
    if replacement.get("error") is not None or replacement.get("stop_reason") == "error":
        raise RuntimeError("Retry result is still an error; original result was not changed.")

    target_indexes = [
        index
        for index, record in enumerate(originals)
        if record.get("test_id") == target_test_id
    ]
    if len(target_indexes) != 1:
        raise RuntimeError("Original smoke results must contain the target exactly once.")
    target_index = target_indexes[0]
    if originals[target_index].get("error") is None:
        raise RuntimeError("Refusing to overwrite a smoke record that is already successful.")
    preserved_successes = {
        record["test_id"]: json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in originals
        if record.get("test_id") != target_test_id and record.get("error") is None
    }
    originals[target_index] = replacement
    originals.sort(key=lambda record: campaign.test_number(record["test_id"]))
    campaign.atomic_write_jsonl(source_path, originals)

    for record in originals:
        test_id = record.get("test_id")
        if test_id in preserved_successes:
            serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
            if serialized != preserved_successes[test_id]:
                raise RuntimeError(f"Successful original record changed unexpectedly: {test_id}")
    return {
        "target_test_id": target_test_id,
        "source_path": str(source_path),
        "retry_path": str(retry_path),
        "source_sha256_before": source_sha256,
        "source_sha256_after": _sha256(source_path),
        "replaced_count": 1,
        "preserved_successful_record_count": len(preserved_successes),
    }


def main() -> int:
    os.chdir(PROJECT_ROOT)
    campaign.load_environment_file(PROJECT_ROOT / ".env")
    keys = campaign.validate_api_keys()
    source_path = campaign.SMOKE_ROOT / TARGET_CONDITION / "results.jsonl"
    source_records = campaign.load_jsonl(source_path)
    target_records = [
        record for record in source_records if record.get("test_id") == TARGET_TEST_ID
    ]
    if len(source_records) != campaign.SMOKE_CASE_COUNT or len(target_records) != 1:
        raise RuntimeError("Smoke source coverage is not the expected 20 cases.")
    if target_records[0].get("error") is None:
        raise RuntimeError(f"{TARGET_TEST_ID} is no longer an error; refusing to rerun it.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retry_root = (
        campaign.SMOKE_ROOT
        / "retries"
        / TARGET_CONDITION
        / TARGET_TEST_ID
        / run_id
    )
    retry_root.mkdir(parents=True, exist_ok=False)
    log_path = retry_root / "run.log"
    environment = os.environ.copy()
    environment["ST_NAV_PROFILE_GEMINI_API_KEY"] = keys[TARGET_CONDITION]
    condition = next(
        condition
        for condition in campaign.CONDITIONS
        if condition["name"] == TARGET_CONDITION
    )
    with log_path.open("w", encoding="utf-8") as log_stream:
        return_code = subprocess.run(
            campaign.evaluation_command(
                test_ids=[TARGET_TEST_ID],
                output_dir=retry_root,
                direction_policy=condition["direction_policy"],
            ),
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        ).returncode
    retry_path = retry_root / "results.jsonl"
    retry_records = campaign.load_jsonl(retry_path)
    retry_succeeded = (
        return_code == 0
        and len(retry_records) == 1
        and retry_records[0].get("test_id") == TARGET_TEST_ID
        and retry_records[0].get("error") is None
        and retry_records[0].get("stop_reason") != "error"
    )
    if not retry_succeeded:
        campaign.atomic_write_json(
            retry_root / "merge_audit.json",
            {
                "status": "retry_failed",
                "target_test_id": TARGET_TEST_ID,
                "evaluator_return_code": return_code,
                "original_changed": False,
                "retry_error": (
                    retry_records[0].get("error") if len(retry_records) == 1 else None
                ),
            },
        )
        return 2

    audit = replace_target_error(
        source_path=source_path,
        retry_path=retry_path,
        target_test_id=TARGET_TEST_ID,
    )
    audit["status"] = "replaced_successfully"
    campaign.atomic_write_json(retry_root / "merge_audit.json", audit)
    print(
        f"[{campaign.utc_now()}] replaced successful retry for {TARGET_TEST_ID}; resuming smoke gate",
        flush=True,
    )
    return campaign.main()


if __name__ == "__main__":
    raise SystemExit(main())
