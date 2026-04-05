from __future__ import annotations


VIEW_DETECTION_KINDS = ("artwork", "landmark", "signage", "passage", "other")
NAVIGATION_TASK_TYPES = (
    "artwork_goal_navigation",
    "artwork_instruction_following_navigation",
    "gallery_goal_navigation",
    "gallery_instruction_following_navigation",
)


def build_view_detection_instructions() -> str:
    return " ".join(
        [
            "You are a museum navigation vision detector for the British Museum.",
            "You are given multiple overlapping views from the same panorama and must reason across all of them together.",
            "Return only museum-relevant entities that are directly visible in at least one provided image.",
            "Prefer stable navigation cues such as famous artifacts, sculptures, reliefs, room signs, distinctive landmarks, and salient passages or doorways.",
            "Before listing other entities, first identify the full set of distinct navigable openings visible across the panorama.",
            "If multiple doorways, arches, corridors, or gallery openings are visible, include each distinct opening as its own entity instead of reporting only the most salient one.",
            "Aggregate duplicate sightings across views only when they are clearly the same physical entity.",
            "Do not merge different entities just because they share the same type, are nearby, or belong to the same architectural area.",
            "If there are multiple passages, doorways, statues, reliefs, or signs, return them as separate entities whenever they correspond to different physical instances.",
            "When needed, disambiguate similar entities with short position-aware labels such as left, right, center, near doorway, or by nearby landmark.",
            "For passages and doorways, treat direction as important identity evidence: a north-facing passage and a south-facing passage should usually be separate entities unless they are unmistakably the exact same opening in overlapping neighboring views.",
            "Do not combine passages seen in opposite or non-contiguous views into one entity.",
            "Do not omit side openings just because they are partially occluded by columns, statues, cases, or wall fragments if the opening is still visibly navigable.",
            "Treat room or gallery labels as signage by default, not as proof that a nearby doorway leads to that labeled room.",
            "Only mention a room id or destination in a passage name when the image directly shows that destination for the passage itself, such as a directional sign or unambiguous doorway label.",
            "Use a specific official exhibit name only when the identity is visually unique and well supported.",
            "When the exact identity is uncertain, return a short descriptive label instead of guessing.",
            "Do not infer unseen objects from room context alone.",
            "If nothing can be identified reliably, return an empty list.",
        ]
    )


def build_view_detection_input(captures: list[dict[str, object]]) -> str:
    labels = []
    for capture in captures:
        label = capture.get("label")
        heading = capture.get("heading")
        if not isinstance(label, str) or not label:
            continue
        if isinstance(heading, (int, float)):
            labels.append(f"{label} ({float(heading):.1f} deg)")
        else:
            labels.append(label)

    lines = [
        f"These are {len(captures)} overlapping views from the same panorama.",
        "Aggregate all visible evidence across the full panorama before answering.",
    ]
    if labels:
        lines.append("Available views: " + ", ".join(labels) + ".")
    lines.extend(
        [
            "Identify only visually grounded entities in these images.",
            "First inventory the distinct navigable openings visible across all views, then report other entities.",
            "Merge evidence across views only for the same physical entity, not for separate instances of the same kind.",
            "For passages, prefer direction-aware names such as north passage, south doorway, east corridor, or doorway beside the Assyria Nimrud sign.",
            "When a room shows several openings around the viewer, include every clearly visible opening even if some are less central or partly occluded.",
            "Only assign multiple source views to one passage when those views are neighboring and show the same opening continuously.",
            "Do not rename a passage as an entrance to Room 8 just because a nearby sign says Assyria Nimrud 8.",
            "For iconic objects with distinctive appearance, a real exhibit name is allowed.",
            "Otherwise use a concise descriptive noun phrase.",
            "Confidence should reflect visual certainty after considering all views together.",
        ]
    )
    return " ".join(lines)


def build_view_detection_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": list(VIEW_DETECTION_KINDS)},
                        "confidence": {"type": "number"},
                        "source_views": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "kind", "confidence", "source_views"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    }


def build_navigation_parse_instructions() -> str:
    return " ".join(
        [
            "You are a museum navigation reasoning parser.",
            "Classify the instruction into exactly one task type.",
            "Extract the source, ordered waypoints, and ordered goals from the instruction.",
            "Preserve entity order from the original instruction.",
            "Treat explicit room or gallery mentions as gallery entities.",
            "Infer artwork room ids using only the provided gallery themes.",
            "Do not invent room ids, entities, or extra steps.",
            "If the source is implicit or missing, return source_room_id as null and source_entity as null.",
            "For gallery entities, predicted_room_id must match the mentioned room id and confidence must be 1.0.",
            "For artwork entities, predicted_room_id must be one allowed room id and confidence must be between 0 and 1.",
        ]
    )


def build_navigation_parse_input(*, instruction: str, room_ids: list[str], theme_lines: str) -> str:
    return "\n".join(
        [
            "Allowed room ids:",
            ", ".join(room_ids),
            "",
            "Gallery themes:",
            theme_lines,
            "",
            "Task:",
            "1. Determine the navigation task type.",
            "2. Extract the source entity if present.",
            "3. Extract goal entities and waypoint entities in order.",
            "4. Ground each entity to one allowed room id or null when required by the schema.",
            "",
            f"Instruction: {instruction}",
        ]
    )


def build_navigation_parse_schema(room_ids: list[str]) -> dict:
    entity_schema = {
        "type": ["object", "null"],
        "properties": {
            "name": {"type": "string"},
            "entity_type": {"type": "string", "enum": ["gallery", "artwork"]},
            "predicted_room_id": {"type": ["string", "null"], "enum": room_ids + [None]},
            "confidence": {"type": "number"},
        },
        "required": ["name", "entity_type", "predicted_room_id", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "task_type": {"type": "string", "enum": list(NAVIGATION_TASK_TYPES)},
            "source_room_id": {"type": ["string", "null"], "enum": room_ids + [None]},
            "source_entity": entity_schema,
            "goal_entities": {"type": "array", "items": entity_schema},
            "waypoint_entities": {"type": "array", "items": entity_schema},
        },
        "required": ["task_type", "source_room_id", "source_entity", "goal_entities", "waypoint_entities"],
        "additionalProperties": False,
    }
