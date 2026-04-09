from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_json,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from st_nav import (
    EntityDetection,
    GroundingIndex,
    LLMRoomLocalizer,
    Observation,
    PerceptionPipeline,
    RoomLocalizer,
    SpatialEngine,
    build_grounding_template,
    load_dotenv,
)
from st_nav_data.normalize import normalize_pano_graph, normalize_room_graph

load_dotenv(PROJECT_ROOT / ".env")

PROBABILITY_DECIMALS = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run room-level localization on synthetic or cached inputs.")
    parser.add_argument("--mode", choices=["synthetic", "manifest", "perception-json"], default="synthetic")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--manifest-path")
    parser.add_argument("--perception-json-path")
    parser.add_argument("--prior-localization-json")
    parser.add_argument("--start-pano-id", default="demo-start-pano")
    parser.add_argument("--start-room-id", default="Room 10")
    parser.add_argument("--current-heading", type=float, default=330.0)
    parser.add_argument("--localizer", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--llm-timeout", type=float, default=30.0)
    parser.add_argument(
        "--prior-room",
        action="append",
        default=[],
        help="Prior room belief in the form Room 10=0.7. Repeatable.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--full-json", action="store_true")
    parser.add_argument("--output-path")
    return parser
def parse_prior_room_belief(values: list[str], default_room_id: str | None) -> dict[str, float]:
    if not values:
        return {default_room_id: 1.0} if default_room_id else {}

    belief: dict[str, float] = {}
    for value in values:
        room_id, sep, probability_text = value.partition("=")
        room_id = room_id.strip()
        if not sep or not room_id:
            raise ValueError(f"Invalid --prior-room value: {value}")
        probability = float(probability_text.strip())
        belief[room_id] = probability

    total = sum(probability for probability in belief.values() if probability > 0.0)
    if total <= 0.0:
        raise ValueError("Prior room belief must contain at least one positive probability.")
    return {room_id: probability / total for room_id, probability in belief.items() if probability > 0.0}


def load_prior_from_localization_json(path: Path) -> tuple[dict[str, float], str | None]:
    payload = load_json(path)
    localizer_payload = payload.get("localizer")
    if not isinstance(localizer_payload, dict):
        raise RuntimeError(f"Localization JSON missing `localizer` block: {path}")

    posterior = localizer_payload.get("posterior_room_belief")
    if not isinstance(posterior, dict):
        raise RuntimeError(f"Localization JSON missing `posterior_room_belief`: {path}")

    prior_room_belief = {
        room_id: float(probability)
        for room_id, probability in posterior.items()
        if isinstance(room_id, str) and isinstance(probability, (int, float)) and probability > 0.0
    }
    if not prior_room_belief:
        raise RuntimeError(f"No positive room belief found in localization JSON: {path}")

    normalized = parse_prior_room_belief(
        [f"{room_id}={probability}" for room_id, probability in prior_room_belief.items()],
        None,
    )
    predicted_room_id = localizer_payload.get("predicted_room_id")
    if not isinstance(predicted_room_id, str) or not predicted_room_id:
        predicted_room_id = None
    return normalized, predicted_room_id


def build_synthetic_demo_inputs() -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], Observation, str]:
    explicit_map = {
        "Room 7": {
            "name": "Room 7",
            "Level": 0,
            "category": "Middle East",
            "title": "Assyria",
            "links": [{"direction": "right", "name": "Room 10"}],
        },
        "Room 10": {
            "name": "Room 10",
            "Level": 0,
            "category": "Middle East",
            "title": "Assyria: Lion hunts",
            "links": [
                {"direction": "left", "name": "Room 7"},
                {"direction": "up", "name": "Room 23"},
            ],
        },
        "Room 18": {
            "name": "Room 18",
            "Level": 0,
            "category": "Ancient Greece and Rome",
            "title": "Greek sculpture",
            "links": [{"direction": "up", "name": "Room 19"}],
        },
        "Room 19": {
            "name": "Room 19",
            "Level": 0,
            "category": "Ancient Greece and Rome",
            "title": "Greek marble sculpture",
            "links": [
                {"direction": "down", "name": "Room 18"},
                {"direction": "up", "name": "Room 20"},
            ],
        },
        "Room 20": {
            "name": "Room 20",
            "Level": 0,
            "category": "Ancient Greece and Rome",
            "title": "Roman sculpture",
            "links": [{"direction": "down", "name": "Room 19"}],
        },
        "Room 23": {
            "name": "Room 23",
            "Level": 0,
            "category": "Ancient Greece and Rome",
            "title": "Greek and Roman sculpture",
            "links": [{"direction": "down", "name": "Room 10"}],
        },
    }
    pano_graph = {
        "demo-start-pano": {
            "panoID": "demo-start-pano",
            "floor": "0",
            "lat": 0.0,
            "lng": 0.0,
            "links": [{"panoID": "demo-current-pano", "heading": 0.0, "description": "towards Room 23"}],
        },
        "demo-current-pano": {
            "panoID": "demo-current-pano",
            "floor": "0",
            "lat": 0.0,
            "lng": 0.0,
            "links": [{"panoID": "demo-start-pano", "heading": 180.0, "description": "towards Room 10"}],
        },
    }

    room_graph = normalize_room_graph(explicit_map, max_room_number=100)
    normalized_pano_graph = normalize_pano_graph(pano_graph)
    grounding = build_grounding_template(room_graph)
    observation = Observation(
        pano_id="demo-current-pano",
        entities=[
            EntityDetection(
                name="Greek Roman statue",
                confidence=0.95,
                kind="artwork",
                source_view="north",
            ),
            EntityDetection(
                name="marble sculpture",
                confidence=0.90,
                kind="landmark",
                source_view="east",
            ),
            EntityDetection(
                name="stone relief",
                confidence=0.75,
                kind="artwork",
                source_view="south",
            ),
        ],
        heading_estimate=0.0,
        metadata={"floor": "0", "demo_case": "thesis_room23"},
    )
    description = (
        "Synthetic thesis-style example: prior room is Room 10, current observation contains "
        "Greek/Roman sculpture evidence, and only Room 7 / Room 10 / Room 23 are reachable from Room 10."
    )
    return room_graph, normalized_pano_graph, grounding, observation, description


def build_manifest_demo_inputs(
    *,
    artifacts_dir: Path,
    manifest_path: Path,
    current_heading: float,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], Observation, str]:
    room_graph = load_json(artifacts_dir / "room_graph.json")
    pano_graph = load_json(artifacts_dir / "pano_graph.json")

    grounding_path = artifacts_dir / "room_grounding.template.json"
    if not grounding_path.exists():
        raise RuntimeError(f"Missing grounding template: {grounding_path}")
    grounding = load_json(grounding_path)

    pipeline = PerceptionPipeline(pano_graph=pano_graph)
    observation = pipeline.observe_from_manifest(manifest_path, current_heading=current_heading)
    description = f"Manifest-based demo from cached detections: {manifest_path}"
    return room_graph, pano_graph, grounding, observation, description


def build_perception_json_demo_inputs(
    *,
    artifacts_dir: Path,
    perception_json_path: Path,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], Observation, str]:
    room_graph = load_json(artifacts_dir / "room_graph.json")
    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    grounding_path = artifacts_dir / "room_grounding.template.json"
    if not grounding_path.exists():
        raise RuntimeError(f"Missing grounding template: {grounding_path}")
    grounding = load_json(grounding_path)

    payload = load_json(perception_json_path)
    manifest_path = payload.get("manifest_path")
    manifest_metadata: dict = {}
    if isinstance(manifest_path, str) and manifest_path:
        manifest_file = Path(manifest_path).resolve()
        if manifest_file.exists():
            manifest_metadata = load_json(manifest_file)

    pano_id = payload.get("pano_id")
    if not isinstance(pano_id, str) or not pano_id:
        raise RuntimeError(f"Missing pano_id in perception JSON: {perception_json_path}")

    entities: list[EntityDetection] = []
    raw_entities = payload.get("entities")
    if isinstance(raw_entities, list):
        for record in raw_entities:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            if not isinstance(name, str) or not name:
                continue
            kind = record.get("kind")
            confidence = record.get("confidence")
            if not isinstance(kind, str) or not kind:
                kind = "other"
            if not isinstance(confidence, (int, float)):
                confidence = 0.0

            source_views = record.get("source_views")
            normalized_source_views: list[str] = []
            if isinstance(source_views, list):
                for value in source_views:
                    if isinstance(value, str) and value and value not in normalized_source_views:
                        normalized_source_views.append(value)

            entities.append(
                EntityDetection(
                    name=name,
                    confidence=float(confidence),
                    kind=kind,
                    source_view=normalized_source_views[0] if len(normalized_source_views) == 1 else "multiview",
                    metadata={
                        "source_views": normalized_source_views,
                        "view_count": len(normalized_source_views),
                    },
                )
            )

    metadata = {
        "manifest_path": manifest_path if isinstance(manifest_path, str) else None,
        "floor": payload.get("floor", manifest_metadata.get("floor")),
        "lat": payload.get("lat", manifest_metadata.get("lat")),
        "lng": payload.get("lng", manifest_metadata.get("lng")),
        "source": "perception-json",
    }
    current_heading = payload.get("current_heading")
    heading_estimate = float(current_heading) if isinstance(current_heading, (int, float)) else None
    observation = Observation(
        pano_id=pano_id,
        entities=entities,
        heading_estimate=heading_estimate,
        metadata=metadata,
    )
    description = f"Perception-JSON demo from: {perception_json_path}"
    return room_graph, pano_graph, grounding, observation, description


def format_belief_lines(title: str, belief: dict[str, float], top_k: int) -> list[str]:
    lines = [title]
    if not belief:
        lines.append("  (empty)")
        return lines
    ordered = sorted(belief.items(), key=lambda item: (-item[1], item[0]))
    for room_id, probability in ordered[: max(top_k, 0)]:
        lines.append(f"  {room_id:<10} {probability:.{PROBABILITY_DECIMALS}f}")
    return lines


def format_entity_lines(observation: Observation) -> list[str]:
    lines = ["Observation Entities"]
    if not observation.entities:
        lines.append("  (none)")
        return lines
    for entity in observation.entities:
        lines.append(
            f"  {entity.name} | kind={entity.kind} | confidence={entity.confidence:.2f} | source={entity.source_view}"
        )
    return lines


def compact_distribution(
    values: dict[str, float],
    *,
    top_k: int | None = None,
    min_value: float = 1e-12,
) -> dict[str, float]:
    filtered = {
        room_id: float(probability)
        for room_id, probability in values.items()
        if isinstance(probability, (int, float)) and float(probability) > min_value
    }
    ordered = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[: max(top_k, 0)]
    return {room_id: probability for room_id, probability in ordered}


def round_probability(value: float | int | None, decimals: int = PROBABILITY_DECIMALS) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), decimals)


def round_distribution(values: dict[str, float], decimals: int = PROBABILITY_DECIMALS) -> dict[str, float]:
    return {
        room_id: round(float(probability), decimals)
        for room_id, probability in values.items()
        if isinstance(probability, (int, float))
    }


def main() -> int:
    args = build_parser().parse_args()
    prior_room_belief = parse_prior_room_belief(args.prior_room, args.start_room_id)
    prior_room_source = "manual"
    if args.prior_localization_json:
        prior_path = resolve_project_path(args.prior_localization_json)
        prior_room_belief, inferred_start_room_id = load_prior_from_localization_json(prior_path)
        prior_room_source = str(prior_path)
        if (not args.start_room_id or args.start_room_id == "Room 10") and inferred_start_room_id:
            args.start_room_id = inferred_start_room_id

    if args.mode == "synthetic":
        room_graph, pano_graph, grounding, observation, description = build_synthetic_demo_inputs()
        start_pano_id = "demo-start-pano"
    elif args.mode == "manifest":
        if not args.manifest_path:
            raise RuntimeError("--manifest-path is required when --mode manifest.")
        artifacts_dir = load_normalized_artifacts(args.artifacts_dir).artifacts_dir
        room_graph, pano_graph, grounding, observation, description = build_manifest_demo_inputs(
            artifacts_dir=artifacts_dir,
            manifest_path=resolve_project_path(args.manifest_path),
            current_heading=args.current_heading,
        )
        start_pano_id = args.start_pano_id
    else:
        if not args.perception_json_path:
            raise RuntimeError("--perception-json-path is required when --mode perception-json.")
        artifacts_dir = load_normalized_artifacts(args.artifacts_dir).artifacts_dir
        room_graph, pano_graph, grounding, observation, description = build_perception_json_demo_inputs(
            artifacts_dir=artifacts_dir,
            perception_json_path=resolve_project_path(args.perception_json_path),
        )
        start_pano_id = args.start_pano_id

    grounding_index = GroundingIndex(grounding)
    if args.localizer == "llm":
        localizer = LLMRoomLocalizer(
            room_graph=room_graph,
            grounding_index=grounding_index,
            model=args.llm_model,
            api_key=args.llm_api_key,
            request_timeout=args.llm_timeout,
        )
    else:
        localizer = RoomLocalizer(
            room_graph=room_graph,
            grounding_index=grounding_index,
        )
    localization = localizer.localize(
        observation=observation,
        prior_room_belief=prior_room_belief,
        fallback_room_id=args.start_room_id,
    )
    if localization.get("predicted_room_id"):
        observation.metadata["localized_room_id"] = localization.get("predicted_room_id")
        observation.metadata["localization_confidence"] = localization.get("confidence", 0.0)
        observation.metadata["room_belief"] = dict(localization.get("room_belief", {}))
        observation.metadata["transition_room_support"] = dict(localization.get("transition_support", {}))
        observation.metadata["observation_room_distribution"] = dict(localization.get("observation_distribution", {}))
        observation.metadata["observation_likelihood"] = dict(localization.get("observation_likelihood", {}))
        observation.metadata["localization_evidence"] = list(localization.get("evidence", []))
        if isinstance(localization.get("summary"), str):
            observation.metadata["localization_summary"] = localization.get("summary")

    spatial = SpatialEngine(
        room_graph=room_graph,
        pano_graph=pano_graph,
        grounding_index=grounding_index,
        localizer=localizer,
    )
    state = spatial.initialize(
        start_pano_id=start_pano_id,
        start_room_id=args.start_room_id,
        start_heading=args.current_heading,
    )
    state.room_belief = dict(prior_room_belief)
    updated_state = spatial.update(state, observation)

    payload = {
        "mode": args.mode,
        "localizer_mode": args.localizer,
        "description": description,
        "start_room_id": args.start_room_id,
        "prior_room_source": prior_room_source,
        "prior_room_belief": prior_room_belief,
        "observation": {
            "pano_id": observation.pano_id,
            "floor": observation.metadata.get("floor"),
            "entity_count": len(observation.entities),
            "entities": [
                {
                    "name": entity.name,
                    "kind": entity.kind,
                    "confidence": entity.confidence,
                    "source_view": entity.source_view,
                }
                for entity in observation.entities
            ],
        },
        "localizer": {
            "predicted_room_id": localization.get("predicted_room_id"),
            "confidence": round_probability(localization.get("confidence")),
            "transition_support": round_distribution(compact_distribution(localization.get("transition_support", {}))),
            "observation_distribution": round_distribution(
                compact_distribution(localization.get("observation_distribution", {}), top_k=args.top_k)
            ),
            "posterior_room_belief": round_distribution(compact_distribution(localization.get("room_belief", {}))),
            "evidence": localization.get("evidence", []),
        },
        "spatial_engine_update": {
            "current_room_id": updated_state.current_room_id,
            "room_belief": round_distribution(compact_distribution(updated_state.room_belief)),
            "observation_metadata": {
                "localized_room_id": observation.metadata.get("localized_room_id"),
                "localization_confidence": round_probability(observation.metadata.get("localization_confidence")),
                "localization_evidence": observation.metadata.get("localization_evidence", []),
            },
        },
    }
    if args.full_json:
        payload["localizer"]["observation_likelihood"] = round_distribution(
            localization.get("observation_likelihood", {})
        )

    output_text = render_json(payload)
    write_text_if_requested(output_text, args.output_path)

    if args.json:
        print(output_text)
        return 0

    lines = [
        "Localization Demo",
        f"Mode: {args.mode}",
        f"Localizer: {args.localizer}",
        f"Description: {description}",
        f"Start room: {args.start_room_id}",
        f"Prior source: {prior_room_source}",
        "",
        *format_belief_lines("Prior Room Belief", prior_room_belief, args.top_k),
        "",
        *format_entity_lines(observation),
        "",
        *format_belief_lines("Transition Support", localization.get("transition_support", {}), args.top_k),
        "",
        *format_belief_lines("Observation Distribution", localization.get("observation_distribution", {}), args.top_k),
        "",
        *format_belief_lines("Posterior Room Belief", localization.get("room_belief", {}), args.top_k),
        "",
        f"Predicted room: {localization.get('predicted_room_id')}",
        f"Localization confidence: {float(localization.get('confidence', 0.0)):.{PROBABILITY_DECIMALS}f}",
        f"Evidence: {', '.join(localization.get('evidence', [])) or '(none)'}",
        "",
        f"SpatialEngine.update() current_room_id: {updated_state.current_room_id}",
        f"SpatialEngine.update() metadata.localized_room_id: {observation.metadata.get('localized_room_id')}",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
