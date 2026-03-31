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
            "Return only museum-relevant entities that are directly visible in the image.",
            "Prefer stable navigation cues such as famous artifacts, sculptures, reliefs, room signs, distinctive landmarks, and salient passages or doorways.",
            "Use a specific official exhibit name only when the identity is visually unique and well supported.",
            "When the exact identity is uncertain, return a short descriptive label instead of guessing.",
            "Do not infer unseen objects from room context alone.",
            "If nothing can be identified reliably, return an empty list.",
        ]
    )


def build_view_detection_input(capture_label: str) -> str:
    return " ".join(
        [
            f"This is the {capture_label} view from a 4-view panorama.",
            "Identify only visually grounded entities in this image.",
            "For iconic objects with distinctive appearance, a real exhibit name is allowed.",
            "Otherwise use a concise descriptive noun phrase.",
            "Confidence should reflect visual certainty, not museum popularity.",
        ]
    )


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
                    },
                    "required": ["name", "kind", "confidence"],
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
