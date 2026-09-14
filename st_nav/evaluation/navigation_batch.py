from __future__ import annotations

from copy import deepcopy
import filecmp
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Sequence

from st_nav.common.model_client import MODEL_TELEMETRY_COUNTER_KEYS
from st_nav.perception.renderer import (
    expected_capture_headings,
    expected_capture_labels,
    manifest_cache_miss_reason,
    sanitize_name,
)


DEFAULT_HEADING_MODE = "museum"
DEFAULT_PITCH = 0.0
DEFAULT_FOV = 90
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
EARTH_RADIUS_M = 6_371_000.0


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
        records.append(payload)
    return records


def compute_step_budget(pano_step_count: int | float) -> int:
    return min(300, max(50, math.ceil(3 * float(pano_step_count))))


def haversine_m(
    lat_a: float,
    lng_a: float,
    lat_b: float,
    lng_b: float,
) -> float:
    """Return the geodesic distance between two WGS84 points in metres."""

    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lng_b - lng_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a)
        * math.cos(phi_b)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def pano_path_distance_m(
    pano_path: Sequence[str],
    pano_graph: dict[str, dict],
) -> float:
    """Sum geodesic metres over every adjacent pano segment in a path."""

    distance = 0.0
    for source_id, target_id in zip(pano_path, pano_path[1:]):
        source = pano_graph.get(source_id)
        target = pano_graph.get(target_id)
        if not isinstance(source, dict) or not isinstance(target, dict):
            raise ValueError(f"Unknown pano segment: {source_id} -> {target_id}")
        coordinates = (
            source.get("lat"),
            source.get("lng"),
            target.get("lat"),
            target.get("lng"),
        )
        if not all(isinstance(value, (int, float)) for value in coordinates):
            raise ValueError(
                f"Missing coordinates for pano segment: {source_id} -> {target_id}"
            )
        distance += haversine_m(*map(float, coordinates))
    return distance


def compute_navigation_spl_metrics(
    *,
    success: bool,
    shortest_path_distance_m: float,
    actual_path_distance_m: float,
    shortest_path_pano_steps: int | None = None,
    movement_steps: int | None = None,
) -> dict:
    """Compute canonical metre-based SPL and an explicitly legacy step variant."""

    shortest_distance = float(shortest_path_distance_m)
    actual_distance = float(actual_path_distance_m)
    if shortest_distance < 0.0 or actual_distance < 0.0:
        raise ValueError("Navigation distances must be non-negative.")
    spl = (
        shortest_distance / max(actual_distance, shortest_distance)
        if success and shortest_distance > 0.0
        else 0.0
    )
    step_spl_legacy = None
    if shortest_path_pano_steps is not None and movement_steps is not None:
        shortest_steps = int(shortest_path_pano_steps)
        actual_steps = int(movement_steps)
        if shortest_steps < 0 or actual_steps < 0:
            raise ValueError("Navigation step counts must be non-negative.")
        step_spl_legacy = (
            shortest_steps / max(shortest_steps, actual_steps)
            if success and shortest_steps > 0
            else 0.0
        )
    return {
        "spl": spl,
        "spl_metric": "distance_m",
        "shortest_path_distance_m": shortest_distance,
        "actual_path_distance_m": actual_distance,
        "actual_over_shortest_ratio": (
            actual_distance / shortest_distance
            if shortest_distance > 0.0
            else None
        ),
        "step_spl_legacy": step_spl_legacy,
    }


def _filter_neighbors(record: dict, *, target_key: str, allowed_ids: set[str]) -> dict:
    filtered = deepcopy(record)
    neighbors = filtered.get("neighbors")
    if isinstance(neighbors, list):
        filtered["neighbors"] = [
            neighbor
            for neighbor in neighbors
            if isinstance(neighbor, dict) and neighbor.get(target_key) in allowed_ids
        ]
    links = filtered.get("links")
    if isinstance(links, list):
        filtered["links"] = [
            link
            for link in links
            if isinstance(link, dict)
            and (link.get("target_pano_id") or link.get("panoID")) in allowed_ids
        ]
    return filtered


def filter_floor_artifacts(
    *,
    pano_graph: dict[str, dict],
    room_graph: dict[str, dict],
    pano_room_grounding: dict,
    floor: str = "0",
) -> tuple[dict[str, dict], dict[str, dict], dict]:
    floor_text = str(floor)
    floor_pano_ids = {
        pano_id
        for pano_id, record in pano_graph.items()
        if isinstance(record, dict) and str(record.get("floor")) == floor_text
    }
    floor_room_ids = {
        room_id
        for room_id, record in room_graph.items()
        if isinstance(record, dict) and str(record.get("floor")) == floor_text
    }
    filtered_panos = {
        pano_id: _filter_neighbors(
            pano_graph[pano_id],
            target_key="target_pano_id",
            allowed_ids=floor_pano_ids,
        )
        for pano_id in sorted(floor_pano_ids)
    }
    filtered_rooms = {
        room_id: _filter_neighbors(
            room_graph[room_id],
            target_key="target_room_id",
            allowed_ids=floor_room_ids,
        )
        for room_id in sorted(floor_room_ids)
    }

    filtered_grounding = deepcopy(pano_room_grounding)
    mappings = pano_room_grounding.get("mappings", pano_room_grounding)
    if not isinstance(mappings, dict):
        mappings = {}
    filtered_mappings = {
        pano_id: room_id
        for pano_id, room_id in mappings.items()
        if pano_id in floor_pano_ids
        and (room_id is None or room_id == "null" or room_id in floor_room_ids)
    }
    if "mappings" in filtered_grounding:
        filtered_grounding["mappings"] = filtered_mappings
    else:
        filtered_grounding = filtered_mappings
    return filtered_panos, filtered_rooms, filtered_grounding


def _load_manifest_pano_id(manifest_path: Path) -> str | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    pano_id = payload.get("pano_id") if isinstance(payload, dict) else None
    return pano_id if isinstance(pano_id, str) and pano_id else None


def _manifest_path(render_root: Path, pano_id: str) -> Path:
    pano_slug = sanitize_name(pano_id)
    return render_root / pano_slug / f"{pano_slug}_manifest.json"


def _valid_manifest_reason(
    manifest_path: Path,
    pano_id: str,
    *,
    heading_mode: str = DEFAULT_HEADING_MODE,
) -> str | None:
    return manifest_cache_miss_reason(
        manifest_path,
        expected_pano_id=pano_id,
        expected_heading_mode=heading_mode,
        expected_pitch=DEFAULT_PITCH,
        expected_fov=DEFAULT_FOV,
        expected_width=DEFAULT_WIDTH,
        expected_height=DEFAULT_HEIGHT,
        expected_labels=expected_capture_labels(heading_mode),
        expected_headings=expected_capture_headings(heading_mode),
    )


def scan_render_root(
    *,
    render_root: str | Path,
    pano_graph: dict[str, dict],
    heading_mode: str = DEFAULT_HEADING_MODE,
) -> dict:
    root = Path(render_root).resolve()
    valid_pano_ids: set[str] = set()
    valid_by_floor: dict[str, int] = {}
    invalid: list[dict] = []
    for manifest_path in sorted(root.glob("*/*_manifest.json")):
        pano_id = _load_manifest_pano_id(manifest_path)
        if pano_id is None:
            invalid.append({"manifest_path": str(manifest_path), "reason": "manifest_invalid"})
            continue
        reason = _valid_manifest_reason(
            manifest_path,
            pano_id,
            heading_mode=heading_mode,
        )
        if reason is not None:
            invalid.append(
                {
                    "pano_id": pano_id,
                    "manifest_path": str(manifest_path),
                    "reason": reason,
                }
            )
            continue
        valid_pano_ids.add(pano_id)
        record = pano_graph.get(pano_id)
        floor = str(record.get("floor")) if isinstance(record, dict) else "unknown"
        valid_by_floor[floor] = valid_by_floor.get(floor, 0) + 1
    return {
        "render_root": str(root),
        "heading_mode": heading_mode,
        "valid_pano_ids": valid_pano_ids,
        "valid_manifest_count": len(valid_pano_ids),
        "valid_by_floor": dict(sorted(valid_by_floor.items())),
        "invalid": invalid,
        "invalid_manifest_count": len(invalid),
    }


def build_cache_inventory(
    *,
    pano_graph: dict[str, dict],
    render_root: str | Path,
    floor: str = "0",
    reference_pano_ids: Iterable[str] = (),
    start_pano_ids: Iterable[str] = (),
    heading_mode: str = DEFAULT_HEADING_MODE,
) -> dict:
    scan = scan_render_root(
        render_root=render_root,
        pano_graph=pano_graph,
        heading_mode=heading_mode,
    )
    valid_pano_ids = scan.pop("valid_pano_ids")
    floor_text = str(floor)
    floor_pano_ids = {
        pano_id
        for pano_id, record in pano_graph.items()
        if isinstance(record, dict) and str(record.get("floor")) == floor_text
    }
    reference_ids = set(reference_pano_ids)
    start_ids = set(start_pano_ids)
    return {
        **scan,
        "floor": floor_text,
        "floor_pano_count": len(floor_pano_ids),
        "floor_cached_count": len(floor_pano_ids & valid_pano_ids),
        "floor_missing_count": len(floor_pano_ids - valid_pano_ids),
        "floor_missing_pano_ids": sorted(floor_pano_ids - valid_pano_ids),
        "reference_pano_count": len(reference_ids),
        "reference_cached_count": len(reference_ids & valid_pano_ids),
        "reference_missing_pano_ids": sorted(reference_ids - valid_pano_ids),
        "start_pano_count": len(start_ids),
        "start_missing_pano_ids": sorted(start_ids - valid_pano_ids),
    }



def _directory_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    if not path.exists():
        return file_count, byte_count
    for candidate in path.rglob("*"):
        if candidate.is_file():
            file_count += 1
            byte_count += candidate.stat().st_size
    return file_count, byte_count


def _rewritten_manifest(
    *,
    source_manifest: dict,
    source_manifest_path: Path,
    destination_root: Path,
    graph_path: Path,
    pano_graph: dict[str, dict],
) -> tuple[dict, list[tuple[Path, Path]]]:
    pano_id = source_manifest["pano_id"]
    pano_slug = sanitize_name(pano_id)
    destination_dir = destination_root / pano_slug
    rewritten = deepcopy(source_manifest)
    rewritten["graph_path"] = str(graph_path.resolve())
    record = pano_graph.get(pano_id, {})
    rewritten["floor"] = str(record.get("floor")) if record.get("floor") is not None else None
    copies: list[tuple[Path, Path]] = []
    captures = []
    for capture in source_manifest["captures"]:
        source_path = Path(capture["path"])
        if not source_path.is_absolute():
            source_path = source_manifest_path.parent / source_path
        destination_path = destination_dir / source_path.name
        rewritten_capture = deepcopy(capture)
        rewritten_capture["path"] = str(destination_path.resolve())
        rewritten_capture.pop("url", None)
        captures.append(rewritten_capture)
        copies.append((source_path, destination_path))
    rewritten["captures"] = captures
    return rewritten, copies


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def _sync_render_pano(
    *,
    pano_id: str,
    source_manifest_path: Path,
    destination_root: Path,
    graph_path: Path,
    pano_graph: dict[str, dict],
    apply: bool,
) -> dict:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    rewritten, copies = _rewritten_manifest(
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        destination_root=destination_root,
        graph_path=graph_path,
        pano_graph=pano_graph,
    )
    destination_dir = destination_root / sanitize_name(pano_id)
    destination_manifest = destination_dir / f"{sanitize_name(pano_id)}_manifest.json"
    desired_text = json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n"
    manifest_changed = (
        not destination_manifest.is_file()
        or destination_manifest.read_text(encoding="utf-8") != desired_text
    )
    images_changed = any(
        not destination_path.is_file()
        or not filecmp.cmp(source_path, destination_path, shallow=False)
        for source_path, destination_path in copies
    )
    changed = manifest_changed or images_changed
    sidecars = [
        sidecar
        for pattern in ("*_detections.json", "*_candidate_themes.json")
        for sidecar in destination_dir.glob(pattern)
    ]
    expected_images = {destination_path.name for _, destination_path in copies}
    obsolete_images = [
        image_path
        for image_path in destination_dir.glob("*.png")
        if image_path.name not in expected_images
    ]

    if apply and changed:
        destination_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{sanitize_name(pano_id)}.sync-",
            dir=destination_root,
        ) as tmpdir:
            temporary_dir = Path(tmpdir)
            staged: list[tuple[Path, Path]] = []
            for source_path, destination_path in copies:
                staged_path = temporary_dir / destination_path.name
                shutil.copy2(source_path, staged_path)
                staged.append((staged_path, destination_path))
            staged_manifest = temporary_dir / destination_manifest.name
            staged_manifest.write_text(desired_text, encoding="utf-8")
            for staged_path, destination_path in staged:
                os.replace(staged_path, destination_path)
            for obsolete_path in obsolete_images:
                obsolete_path.unlink()
            for sidecar_path in sidecars:
                sidecar_path.unlink()
            os.replace(staged_manifest, destination_manifest)

    return {
        "pano_id": pano_id,
        "changed": changed,
        "images_changed": images_changed,
        "sidecars_invalidated": len(sidecars) if apply and changed else 0,
        "obsolete_images_removed": len(obsolete_images) if apply and changed else 0,
    }


def sync_floor0_assets(
    *,
    dataset_source: str | Path,
    dataset_destination: str | Path,
    render_source: str | Path,
    render_destination: str | Path,
    pano_graph: dict[str, dict],
    graph_path: str | Path,
    floor: str = "0",
    apply: bool = False,
) -> dict:
    dataset_source = Path(dataset_source).resolve()
    dataset_destination = Path(dataset_destination).resolve()
    render_source = Path(render_source).resolve()
    render_destination = Path(render_destination).resolve()
    graph_path = Path(graph_path).resolve()
    if not dataset_source.is_dir():
        raise FileNotFoundError(f"Dataset source not found: {dataset_source}")
    if not render_source.is_dir():
        raise FileNotFoundError(f"Render source not found: {render_source}")

    source_scan = scan_render_root(render_root=render_source, pano_graph=pano_graph)
    source_valid = source_scan.pop("valid_pano_ids")
    floor_text = str(floor)
    selected = sorted(
        pano_id
        for pano_id in source_valid
        if isinstance(pano_graph.get(pano_id), dict)
        and str(pano_graph[pano_id].get("floor")) == floor_text
    )
    skipped_non_floor = sorted(source_valid - set(selected))
    source_dataset_files, source_dataset_bytes = _directory_stats(dataset_source)

    if apply:
        dataset_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dataset_source, dataset_destination, dirs_exist_ok=True)
        render_destination.mkdir(parents=True, exist_ok=True)

    sync_records = []
    for pano_id in selected:
        source_manifest_path = _manifest_path(render_source, pano_id)
        sync_records.append(
            _sync_render_pano(
                pano_id=pano_id,
                source_manifest_path=source_manifest_path,
                destination_root=render_destination,
                graph_path=graph_path,
                pano_graph=pano_graph,
                apply=apply,
            )
        )

    floor_pano_ids = {
        pano_id
        for pano_id, record in pano_graph.items()
        if isinstance(record, dict) and str(record.get("floor")) == floor_text
    }
    report = {
        "mode": "apply" if apply else "dry_run",
        "floor": floor_text,
        "dataset": {
            "source": str(dataset_source),
            "destination": str(dataset_destination),
            "source_file_count": source_dataset_files,
            "source_bytes": source_dataset_bytes,
        },
        "renders": {
            "source": str(render_source),
            "destination": str(render_destination),
            "source_valid_manifest_count": len(source_valid),
            "source_invalid_manifest_count": source_scan["invalid_manifest_count"],
            "selected_floor_manifest_count": len(selected),
            "skipped_non_floor_manifest_count": len(skipped_non_floor),
            "floor_graph_pano_count": len(floor_pano_ids),
            "floor_missing_after_source_sync_count": len(floor_pano_ids - set(selected)),
            "floor_missing_after_source_sync_pano_ids": sorted(floor_pano_ids - set(selected)),
            "changed_pano_count": sum(record["changed"] for record in sync_records),
            "sidecars_invalidated": sum(record["sidecars_invalidated"] for record in sync_records),
            "obsolete_images_removed": sum(record["obsolete_images_removed"] for record in sync_records),
        },
    }
    if apply:
        _atomic_write_json(render_destination / "sync_report.json", report)
    return report


def summarize_batch_results(results: list[dict]) -> dict:
    total = len(results)
    successes = sum(bool(record.get("success")) for record in results)
    spl_values = [float(record.get("spl", 0.0)) for record in results]
    spl_metric_values = {
        str(record["spl_metric"])
        for record in results
        if isinstance(record.get("spl_metric"), str)
    }
    step_spl_legacy_values = [
        float(record["step_spl_legacy"])
        for record in results
        if isinstance(record.get("step_spl_legacy"), (int, float))
    ]
    errors = sum(bool(record.get("error")) for record in results)
    model_call_stages: dict[str, dict[str, int]] = {}
    model_call_totals = {key: 0 for key in MODEL_TELEMETRY_COUNTER_KEYS}
    model_call_coverage_count = 0
    for record in results:
        model_calls = record.get("model_calls")
        if not isinstance(model_calls, dict):
            continue
        stages = model_calls.get("stages")
        if not isinstance(stages, dict):
            continue
        model_call_coverage_count += 1
        for stage, counters in stages.items():
            if not isinstance(stage, str) or not isinstance(counters, dict):
                continue
            stage_totals = model_call_stages.setdefault(
                stage,
                {key: 0 for key in MODEL_TELEMETRY_COUNTER_KEYS},
            )
            for key in MODEL_TELEMETRY_COUNTER_KEYS:
                value = int(counters.get(key, 0))
                stage_totals[key] += value
                model_call_totals[key] += value
    return {
        "case_count": total,
        "success_count": successes,
        "success_rate": successes / total if total else 0.0,
        "average_spl": sum(spl_values) / total if total else 0.0,
        "spl_metric": (
            next(iter(spl_metric_values))
            if len(spl_metric_values) == 1
            else "mixed" if spl_metric_values else None
        ),
        "average_step_spl_legacy": (
            sum(step_spl_legacy_values) / len(step_spl_legacy_values)
            if step_spl_legacy_values
            else None
        ),
        "step_spl_legacy_coverage_count": len(step_spl_legacy_values),
        "error_count": errors,
        "total_movement_steps": sum(int(record.get("movement_steps", 0)) for record in results),
        "render": {
            "cache_hit_events": sum(
                int(record.get("render", {}).get("cache_hit_events", 0))
                for record in results
            ),
            "cache_miss_events": sum(
                int(record.get("render", {}).get("cache_miss_events", 0))
                for record in results
            ),
            "new_render_pano_count": sum(
                int(record.get("render", {}).get("new_render_pano_count", 0))
                for record in results
            ),
            "new_image_requests": sum(
                int(record.get("render", {}).get("new_image_requests", 0))
                for record in results
            ),
        },
        "model_calls": {
            "coverage_case_count": model_call_coverage_count,
            "missing_telemetry_case_count": total - model_call_coverage_count,
            "stages": model_call_stages,
            "total": model_call_totals,
        },
    }
