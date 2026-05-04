from __future__ import annotations


VIEW_DETECTION_KINDS = ("artwork", "landmark", "signage", "passage", "other")
NAVIGATION_TASK_TYPES = (
    "artwork_goal_navigation",
    "artwork_instruction_following_navigation",
    "gallery_goal_navigation",
    "gallery_instruction_following_navigation",
)
ALLOCENTRIC_DIRECTIONS = ("north", "east", "south", "west")


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


def build_localization_instructions() -> str:
    return " ".join(
        [
            "You are a museum localization reasoner for British Museum indoor navigation.",
            "You are given detected entities from the current panorama, plus a closed list of candidate rooms on the same floor.",
            "Estimate how visually compatible the current observation is with each candidate room.",
            "Use the room titles, aliases, categories, and anchor entities as room descriptions.",
            "Prefer stable in-room evidence such as statues, reliefs, busts, monuments, display cases, and explicit room signs.",
            "Treat passages, doorways, and corridor views as weaker evidence because they may reveal adjacent rooms.",
            "A sign for another room can indicate a nearby neighboring room, so do not over-weight it unless it strongly matches the overall scene.",
            "Return a normalized observation distribution over the candidate rooms, where probabilities sum to 1.",
            "These probabilities represent observation-only compatibility, not motion priors and not the final localization posterior.",
            "Do not invent room ids outside the candidate list.",
            "Return JSON only.",
        ]
    )


def build_localization_input(*, observation_entities: list[dict], candidates: list[dict]) -> str:
    entity_lines = []
    for entity in observation_entities:
        parts = [
            f"name={entity['name']}",
            f"kind={entity['kind']}",
            f"confidence={entity['confidence']:.2f}",
        ]
        source_views = entity.get("source_views")
        if isinstance(source_views, list) and source_views:
            parts.append("source_views=" + ", ".join(str(value) for value in source_views))
        entity_lines.append("- " + " | ".join(parts))

    candidate_lines = []
    for candidate in candidates:
        parts = [
            f"room_id={candidate['room_id']}",
            f"title={candidate.get('title') or 'unknown'}",
            f"category={candidate.get('category') or 'unknown'}",
        ]
        aliases = candidate.get("aliases")
        if isinstance(aliases, list) and aliases:
            parts.append("aliases=" + ", ".join(str(value) for value in aliases))
        anchor_entities = candidate.get("anchor_entities")
        if isinstance(anchor_entities, list) and anchor_entities:
            parts.append("anchors=" + ", ".join(str(value) for value in anchor_entities))
        candidate_lines.append("- " + " | ".join(parts))

    lines = [
        f"Observed entity count: {len(observation_entities)}.",
        "Observation entities:",
        *entity_lines,
        "",
        f"Candidate room count: {len(candidates)}.",
        "Candidate rooms:",
        *candidate_lines,
        "",
        "Task:",
        "1. Judge visual compatibility for every candidate room using only the observation entities.",
        "2. Output exactly one observation probability for each candidate room.",
        "3. Make the observation probabilities sum to 1 across the candidate rooms.",
        "4. Choose the single best predicted_room_id based on observation compatibility only.",
    ]
    return "\n".join(lines)


def build_localization_schema(room_ids: list[str]) -> dict:
    room_score_schema = {
        "type": "object",
        "properties": {
            "room_id": {"type": "string", "enum": room_ids},
            "score": {"type": "number"},
        },
        "required": ["room_id", "score"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "predicted_room_id": {"type": ["string", "null"], "enum": room_ids + [None]},
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "room_distribution": {"type": "array", "items": room_score_schema},
            "summary": {"type": "string"},
        },
        "required": ["predicted_room_id", "confidence", "evidence", "room_distribution", "summary"],
        "additionalProperties": False,
    }


def build_spatial_context_extraction_instructions() -> str:
    return " ".join(
        [
            "You are a museum panorama analyzer for indoor localization.",
            "You are given several images captured from one panorama in clockwise order.",
            "The global orientation is unknown, so do not assume any image is north, front, right, rear, or left.",
            "Treat each image only as a panorama sector such as view_0 or view_1.",
            "For each view, identify the most relevant exhibit themes or gallery themes visible in that sector.",
            "A single sector may legitimately show multiple adjacent gallery themes at once.",
            "Keep side glimpses into neighboring rooms or cross-gallery openings when they are visually present.",
            "Do not collapse the answer to only one dominant theme if two or three distinct themes are simultaneously visible.",
            "Use concise theme labels grounded in the images, such as Assyria: Nimrud or Greek and Roman sculpture.",
            "If the exact exhibit identity is uncertain, return the closest high-level theme rather than guessing a specific object.",
            "Return JSON only.",
        ]
    )


def build_spatial_context_extraction_input(*, view_ids: list[str], candidate_theme_labels: list[str]) -> str:
    lines = [
        f"Panorama sectors: {', '.join(view_ids)}.",
        "The sectors are listed in clockwise order around the same panorama.",
        "The global heading of the panorama is unknown.",
        "",
        "Known gallery or exhibit themes that may appear:",
    ]
    lines.extend(f"- {label}" for label in candidate_theme_labels)
    lines.extend(
        [
            "",
            "Task:",
            "1. For each panorama sector, list the most relevant visible themes, including multiple simultaneous themes when clearly visible.",
            "2. Provide confidence values between 0 and 1.",
            "3. Keep the answer grounded in what is visible in that sector.",
            "4. If a sector contains views into adjacent galleries, include those adjacent gallery themes instead of suppressing them.",
        ]
    )
    return "\n".join(lines)


def build_spatial_context_extraction_schema(view_ids: list[str]) -> dict:
    theme_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    view_schema = {
        "type": "object",
        "properties": {
            "view_id": {"type": "string", "enum": view_ids},
            "themes": {"type": "array", "items": theme_schema},
            "summary": {"type": "string"},
        },
        "required": ["view_id", "themes", "summary"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "views": {"type": "array", "items": view_schema},
            "summary": {"type": "string"},
        },
        "required": ["views", "summary"],
        "additionalProperties": False,
    }


def build_spatial_alignment_instructions(*, direct_images: bool = False) -> str:
    base_parts = [
        "You are a museum spatial alignment reasoner.",
        "You are given candidate room spatial contexts from the museum map and either panorama-sector text summaries or the panorama images themselves.",
        "The panorama heading is unknown.",
        "Reason over rotation: determine which candidate room best matches the observed panorama after allowing a global rotation offset.",
        "Also infer which allocentric direction the first panorama sector view_0 most likely corresponds to.",
        "Use only the provided candidate rooms.",
    ]
    if direct_images:
        base_parts.extend(
            [
                "When panorama images are provided directly, do not jump straight to a room prediction from one sign or one iconic object.",
                "First infer the dominant visible theme of each relevant panorama sector.",
                "Then align at least two sectors to allocentric directions and nearby candidate-room themes before deciding the room.",
                "Your evidence and alignment trace must explicitly reference sector ids such as view_2 or view_5.",
            ]
        )
    base_parts.append("Return JSON only.")
    return " ".join(
        base_parts
    )


def build_spatial_alignment_input(
    *,
    candidate_context_text: str,
    ego_context_text: str,
    view_ids: list[str],
    direct_images: bool = False,
) -> str:
    lines = [
        "Candidate room spatial contexts:",
        candidate_context_text,
        "",
        "Observed panorama sectors:",
        f"The panorama sectors are ordered clockwise as: {', '.join(view_ids)}.",
        "The global heading is unknown, so the sector labels are not north, east, south, or west.",
        ego_context_text,
        "",
        "Task:",
        "1. Decide which candidate room is most consistent with the observed panorama after allowing rotation.",
        "2. Infer which allocentric direction view_0 most likely corresponds to.",
        "3. Output a normalized room distribution over the candidate rooms.",
        "4. Briefly summarize the main supporting and conflicting evidence.",
    ]
    if direct_images:
        lines.extend(
            [
                "5. Before choosing the room, determine which sectors provide the strongest directional clues.",
                "6. Output a sector_alignment trace that maps at least two panorama sectors to allocentric directions and the candidate-room theme or neighbor they best match.",
                "7. Do not rely only on one room sign if the wider spatial pattern points elsewhere.",
            ]
        )
    return "\n".join(lines)


def build_spatial_alignment_schema(
    room_ids: list[str],
    *,
    view_ids: list[str] | None = None,
    include_sector_alignment: bool = False,
) -> dict:
    room_score_schema = {
        "type": "object",
        "properties": {
            "room_id": {"type": "string", "enum": room_ids},
            "score": {"type": "number"},
        },
        "required": ["room_id", "score"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "predicted_room_id": {"type": ["string", "null"], "enum": room_ids + [None]},
            "confidence": {"type": "number"},
            "view_0_allocentric_direction": {"type": ["string", "null"], "enum": list(ALLOCENTRIC_DIRECTIONS) + [None]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "room_distribution": {"type": "array", "items": room_score_schema},
            "summary": {"type": "string"},
        },
        "required": [
            "predicted_room_id",
            "confidence",
            "view_0_allocentric_direction",
            "evidence",
            "room_distribution",
            "summary",
        ],
        "additionalProperties": False,
    }
    if include_sector_alignment:
        sector_schema = {
            "type": "object",
            "properties": {
                "view_id": {"type": "string", "enum": list(view_ids or [])},
                "allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
                "matched_room_id": {"type": ["string", "null"], "enum": room_ids + [None]},
                "matched_theme": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["view_id", "allocentric_direction", "matched_room_id", "matched_theme", "rationale"],
            "additionalProperties": False,
        }
        schema["properties"]["sector_alignment"] = {"type": "array", "items": sector_schema}
        schema["required"].append("sector_alignment")
    return schema
