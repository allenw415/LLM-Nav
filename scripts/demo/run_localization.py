from __future__ import annotations

import argparse
import json
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
    LLMSpatialAlignmentLocalizer,
    Observation,
    PerceptionPipeline,
    RenderedView,
    RoomLocalizer,
    SpatialEngine,
    build_grounding_template,
    load_dotenv,
    resolve_model_environment,
)
from st_nav_data.normalize import normalize_pano_graph, normalize_room_graph

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)

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
    parser.add_argument(
        "--localizer",
        choices=["bayesian-filter", "heuristic", "llm", "spatial-alignment-a", "spatial-alignment-b", "split-independent"],
        default="bayesian-filter",
    )
    parser.add_argument("--bayesian-localizer", choices=["heuristic", "llm"], default="llm")
    parser.add_argument("--spatial-alignment-mode", choices=["a", "b"], default="a")
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 30.0)
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
    views = [
        RenderedView(
            label=f"view_{index}",
            heading=float(capture.get("heading", index * 90.0)),
            path=str(capture["path"]),
            url=capture.get("url"),
        )
        for index, capture in enumerate(manifest_metadata.get("captures", []))
        if isinstance(capture, dict) and isinstance(capture.get("path"), str) and capture.get("path")
    ]
    current_heading = payload.get("current_heading")
    heading_estimate = float(current_heading) if isinstance(current_heading, (int, float)) else None
    observation = Observation(
        pano_id=pano_id,
        views=views,
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


def compact_spatial_alignment(spatial_alignment: object, *, include_details: bool) -> dict | None:
    if not isinstance(spatial_alignment, dict):
        return None
    compact = {
        "mode": spatial_alignment.get("mode"),
        "view_0_allocentric_direction": spatial_alignment.get("view_0_allocentric_direction"),
    }
    if include_details:
        for key in ("candidate_context_text", "ego_context_text", "ego_context_views"):
            if key in spatial_alignment:
                compact[key] = spatial_alignment.get(key)
    return compact


def compact_ego_spatial_context(ego_spatial_context: object, *, include_details: bool) -> dict | None:
    if not isinstance(ego_spatial_context, dict):
        return None
    compact = {
        "summary": ego_spatial_context.get("summary"),
        "text": ego_spatial_context.get("text"),
    }
    if include_details:
        compact["views"] = ego_spatial_context.get("views")
    return compact


def best_room_from_distribution(distribution: dict[str, float]) -> tuple[str | None, float]:
    ordered = [
        (room_id, float(probability))
        for room_id, probability in distribution.items()
        if isinstance(room_id, str) and isinstance(probability, (int, float))
    ]
    if not ordered:
        return None, 0.0
    room_id, probability = max(ordered, key=lambda item: (item[1], item[0]))
    return room_id, probability


def build_localizer_summary(
    localization: dict,
    *,
    top_k: int,
    full_json: bool,
    observation_only: bool = False,
) -> dict:
    observation_distribution = round_distribution(
        compact_distribution(localization.get("observation_distribution", {}), top_k=top_k)
    )
    posterior_room_belief = round_distribution(compact_distribution(localization.get("room_belief", {})))
    predicted_room_id = localization.get("predicted_room_id")
    confidence = round_probability(localization.get("confidence"))
    if observation_only:
        predicted_room_id, probability = best_room_from_distribution(localization.get("observation_distribution", {}))
        confidence = round_probability(probability)
        posterior_room_belief = observation_distribution

    payload = {
        "predicted_room_id": predicted_room_id,
        "confidence": confidence,
        "transition_support": round_distribution(compact_distribution(localization.get("transition_support", {}))),
        "observation_distribution": observation_distribution,
        "posterior_room_belief": posterior_room_belief,
        "evidence": localization.get("evidence", []),
        "spatial_alignment": compact_spatial_alignment(
            localization.get("spatial_alignment"),
            include_details=full_json,
        ),
        "ego_spatial_context": compact_ego_spatial_context(
            localization.get("ego_spatial_context"),
            include_details=full_json,
        ),
    }
    if full_json:
        payload["observation_likelihood"] = round_distribution(localization.get("observation_likelihood", {}))
    return payload


def build_spatial_alignment_summary(
    localization: dict,
    *,
    top_k: int,
    full_json: bool,
) -> dict:
    spatial_alignment = localization.get("spatial_alignment")
    ego_spatial_context = localization.get("ego_spatial_context")
    map_spatial_context = None
    inferred_direction = None
    if isinstance(spatial_alignment, dict):
        map_spatial_context = spatial_alignment.get("candidate_context_text")
        inferred_direction = spatial_alignment.get("view_0_allocentric_direction")

    ego_payload = None
    if isinstance(ego_spatial_context, dict):
        ego_payload = {
            "summary": ego_spatial_context.get("summary"),
            "text": ego_spatial_context.get("text"),
        }
        if full_json:
            ego_payload["views"] = ego_spatial_context.get("views")

    return {
        "predicted_room_id": localization.get("predicted_room_id"),
        "confidence": round_probability(localization.get("confidence")),
        "observation_distribution": round_distribution(
            compact_distribution(localization.get("observation_distribution", {}), top_k=top_k)
        ),
        "evidence": localization.get("evidence", []),
        "map_spatial_context": map_spatial_context,
        "ego_spatial_context": ego_payload,
        "inferred_view_0_allocentric_direction": inferred_direction,
    }


def extract_room_context_block(map_spatial_context: object, room_id: str | None) -> str | None:
    if not isinstance(map_spatial_context, str) or not map_spatial_context or not isinstance(room_id, str) or not room_id:
        return None
    lines = map_spatial_context.splitlines()
    target_header = f"Candidate room {room_id}:"
    start_index = None
    for index, line in enumerate(lines):
        if line.startswith(target_header):
            start_index = index
            break
    if start_index is None:
        return None

    block = []
    for line in lines[start_index:]:
        if block and line.startswith("Candidate room "):
            break
        block.append(line)
    return "\n".join(block).strip() or None


def append_multiline_block(lines: list[str], title: str, body: str | None) -> None:
    lines.append(title)
    if not body:
        lines.append("  (none)")
        return
    for raw_line in body.splitlines():
        lines.append(f"  {raw_line}" if raw_line else "  ")


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

    def build_bayesian_localizer(kind: str):
        if kind == "llm":
            return LLMRoomLocalizer(
                room_graph=room_graph,
                grounding_index=grounding_index,
                model=args.llm_model,
                api_key=args.llm_api_key,
                api_base=args.llm_api_base,
                api_kind=args.llm_api_kind,
                request_timeout=args.llm_timeout,
            )
        return RoomLocalizer(
            room_graph=room_graph,
            grounding_index=grounding_index,
        )

    def build_spatial_alignment_localizer(mode: str):
        return LLMSpatialAlignmentLocalizer(
            room_graph=room_graph,
            grounding_index=grounding_index,
            alignment_mode="text_from_images" if mode == "a" else "direct_images",
            model=args.llm_model,
            api_key=args.llm_api_key,
            api_base=args.llm_api_base,
            api_kind=args.llm_api_kind,
            request_timeout=args.llm_timeout,
        )

    observation_only = False
    if args.localizer == "bayesian-filter":
        localizer = build_bayesian_localizer(args.bayesian_localizer or "llm")
    elif args.localizer == "heuristic":
        localizer = build_bayesian_localizer("heuristic")
    elif args.localizer == "llm":
        localizer = build_bayesian_localizer("llm")
    elif args.localizer == "spatial-alignment-a":
        localizer = build_spatial_alignment_localizer("a")
        observation_only = True
    elif args.localizer == "spatial-alignment-b":
        localizer = build_spatial_alignment_localizer("b")
        observation_only = True
    elif args.localizer == "split-independent":
        raise RuntimeError(
            "Use `--localizer bayesian-filter` or `--localizer spatial-alignment-a|b`. "
            "`split-independent` is no longer the recommended mode."
        )
    else:
        localizer = build_bayesian_localizer("heuristic")

    if args.localizer in {"spatial-alignment-a", "spatial-alignment-b"} and not observation.views:
        raise RuntimeError(
            f"{args.localizer} requires panorama images. Use --mode manifest or a perception JSON that includes manifest captures."
        )
    localization = localizer.localize(
        observation=observation,
        prior_room_belief=prior_room_belief,
        fallback_room_id=args.start_room_id,
    )
    if args.localizer in {"spatial-alignment-a", "spatial-alignment-b"}:
        summary = build_spatial_alignment_summary(
            localization,
            top_k=args.top_k,
            full_json=args.full_json,
        )
    else:
        summary = build_localizer_summary(
            localization,
            top_k=args.top_k,
            full_json=args.full_json,
            observation_only=observation_only,
        )

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
        "localizer": summary,
    }

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
    ]
    if args.localizer in {"spatial-alignment-a", "spatial-alignment-b"}:
        ego_spatial_context = summary.get("ego_spatial_context") if isinstance(summary.get("ego_spatial_context"), dict) else {}
        ego_context_text = ego_spatial_context.get("text") if isinstance(ego_spatial_context, dict) else None
        map_context_block = summary.get("map_spatial_context")
        lines.extend(
            [
                *format_belief_lines("Observation Distribution", summary.get("observation_distribution", {}), args.top_k),
                "",
                f"Predicted room: {summary.get('predicted_room_id')}",
                f"Localization confidence: {float(summary.get('confidence') or 0.0):.{PROBABILITY_DECIMALS}f}",
                (
                    "Inferred view_0 allocentric direction: "
                    f"{summary.get('inferred_view_0_allocentric_direction')}"
                    if summary.get("inferred_view_0_allocentric_direction")
                    else "Inferred view_0 allocentric direction: (none)"
                ),
                f"Evidence: {', '.join(summary.get('evidence', [])) or '(none)'}",
            ]
        )
        lines.append("")
        append_multiline_block(lines, "Map Spatial Context", map_context_block)
        lines.append("")
        append_multiline_block(lines, "Ego Spatial Context", ego_context_text)
    else:
        lines.extend(
            [
                *format_belief_lines("Transition Support", summary.get("transition_support", {}), args.top_k),
                "",
                *format_belief_lines("Observation Distribution", summary.get("observation_distribution", {}), args.top_k),
                "",
                *format_belief_lines("Posterior Room Belief", summary.get("posterior_room_belief", {}), args.top_k),
                "",
                f"Predicted room: {summary.get('predicted_room_id')}",
                f"Localization confidence: {float(summary.get('confidence') or 0.0):.{PROBABILITY_DECIMALS}f}",
                f"Evidence: {', '.join(summary.get('evidence', [])) or '(none)'}",
            ]
        )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
