from __future__ import annotations


VIEW_DETECTION_KINDS = ("artwork", "landmark", "signage", "passage", "other")
ENTITY_LOCATION_SCOPES = ("inside", "outside", "unknown")
ROOM_EVIDENCE_TYPES = ("weak_generic", "shared_theme", "room_specific", "direct_room_label")
NAVIGATION_TASK_TYPES = (
    "artwork_gallery_instruction_following_navigation",
    "artwork_goal_navigation",
    "artwork_instruction_following_navigation",
    "gallery_goal_navigation",
    "gallery_instruction_following_navigation",
)
ALLOCENTRIC_DIRECTIONS = ("north", "east", "south", "west")
CANONICAL_ROOM_THEMES = (
    ("Room 4", "Ancient Egypt"),
    ("Room 6", "Assyrian sculpture and Balawat Gates"),
    ("Rooms 7, 8", "Assyria: Nimrud"),
    ("Room 9", "Assyria: Nineveh"),
    ("Room 10", "Assyria: Lion hunts, Siege of Lachish and Khorsabad"),
    ("Room 12", "Greece: Minoans and Mycenaeans"),
    ("Room 13", "Greece 1050-520 BC"),
    ("Room 14", "Greek vases"),
    ("Room 15", "Athens and Lycia"),
    ("Room 16", "Greece: Bassai sculptures"),
    ("Room 17", "Nereid Monument"),
    ("Room 18", "Greece: Parthenon"),
    ("Room 19", "Greece: Athens"),
    ("Room 20", "Greeks and Lycians 400-325 BC"),
    ("Room 21", "Mausoleum of Halikarnassos"),
    ("Room 22", "The World of Alexander"),
    ("Room 23", "Greek and Roman sculpture"),
)


def _append_visual_profile_parts(parts: list[str], candidate: dict) -> None:
    short_description = candidate.get("short_description")
    if isinstance(short_description, str) and short_description:
        parts.append(f"short_description={short_description}")
    visual_cues = candidate.get("visual_cues")
    if isinstance(visual_cues, list) and visual_cues:
        parts.append("visual_cues=" + ", ".join(str(value) for value in visual_cues))


def canonical_room_themes_text() -> str:
    return "\n".join(f"- {room_id}: {theme}" for room_id, theme in CANONICAL_ROOM_THEMES)


def build_view_theme_extraction_instructions() -> str:
    return " ".join(
        [
            "You extract visual observations from museum panorama sectors.",
            "Do not localize the camera or choose the current room.",
            "Room candidates and the museum map are intentionally hidden.",
            "For each sector, identify visible evidence, visible room or gallery label if any, best matching canonical room themes, and whether the evidence appears current, adjacent, both, or ambiguous.",
            "Use only the provided canonical room themes; do not invent new theme labels.",
            "Return JSON only.",
        ]
    )


def build_view_theme_extraction_input(captures: list[dict[str, object]]) -> str:
    view_ids = [f"view_{index}" for index, _ in enumerate(captures)]
    return "\n".join(
        [
            "Per-view theme extraction task:",
            f"These are {len(captures)} overlapping sectors from the same panorama.",
            "The sectors are ordered clockwise, but the absolute allocentric direction of view_0 is unknown.",
            "Inspect each sector independently.",
            "Do not use room candidates, map relations, or heading labels.",
            "Use confidence from 0.0 to 1.0 for each canonical theme match.",
            "If evidence is generic or could match multiple themes, return multiple low-confidence theme_matches.",
            'Use current_or_adjacent="ambiguous" when it is unclear whether the evidence is current-room or adjacent-room.',
            "Include one view_theme_observations record for each view when possible.",
            "",
            "Canonical room themes:",
            canonical_room_themes_text(),
            "",
            f"Panorama sectors in clockwise order: {', '.join(view_ids)}.",
        ]
    )


def build_view_theme_extraction_schema(view_ids: list[str]) -> dict:
    observation_schema = {
        "type": "object",
        "properties": {
            "view_id": {"type": "string", "enum": view_ids},
            "observed_theme": {
                "type": "string",
                "description": "Best matching canonical theme, or none. Do not invent labels.",
            },
            "confidence": {"type": "number"},
            "visible_room_label": {"type": ["string", "null"]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "visual_evidence": {"type": "array", "items": {"type": "string"}},
            "theme_matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "room_ids": {"type": "array", "items": {"type": "string"}},
                        "canonical_theme": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string", "maxLength": 160},
                    },
                    "required": ["room_ids", "canonical_theme", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
            "current_or_adjacent": {"type": "string", "enum": ["current", "adjacent", "both", "ambiguous"]},
            "spatial_boundary_evidence": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string", "maxLength": 160},
        },
        "required": [
            "view_id",
            "observed_theme",
            "confidence",
            "visible_room_label",
            "evidence",
            "visual_evidence",
            "theme_matches",
            "current_or_adjacent",
            "spatial_boundary_evidence",
            "reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "view_theme_observations": {"type": "array", "items": observation_schema},
            "summary": {"type": "string"},
        },
        "required": ["view_theme_observations", "summary"],
        "additionalProperties": False,
    }


def build_visual_detection_localization_instructions() -> str:
    return " ".join(
        [
            "You are a museum panorama vision localizer for British Museum indoor navigation.",
            "You are given multiple overlapping views from the same panorama and a closed list of candidate rooms.",
            "Reason internally before answering, but return only the requested JSON.",
            "First identify visible museum-relevant entities across all views.",
            "Merge repeated sightings of the same entity across views into one entity with combined source_views.",
            "Return at most eight distinct entities, prioritizing the most discriminative visible evidence.",
            "Classify each entity as inside when it belongs to the current gallery, outside when it is only visible through a doorway, corridor, or adjacent-gallery opening, and unknown when the scope is visually ambiguous.",
            "Keep outside entities in the entity list, but use inside entities as the primary evidence for localization.",
            "Use candidate room titles, aliases, categories, and anchor entities as gallery theme descriptions.",
            "Treat passages, doorway views, and signs for adjacent rooms as useful context but weak evidence for the current room.",
            "Return independent evidence scores for every candidate room, not probabilities.",
            "Before scoring each candidate room, classify the strongest supporting evidence as weak_generic, shared_theme, room_specific, or direct_room_label.",
            "weak_generic means common visual evidence that appears in many museum rooms, such as display cases, statues, stone objects, doorways, arches, wall reliefs, or generic gallery layout.",
            "shared_theme means evidence matching a broad theme shared by multiple candidate rooms. For example, Assyrian reliefs, Lamassu, Assyrian sculpture, or Middle East gallery can support several nearby Assyrian rooms such as Rooms 7, 8, 9, and 10.",
            "room_specific means evidence pointing to a specific room theme or landmark, but not a directly visible room label.",
            "direct_room_label means directly visible and legible text that explicitly identifies the current room or gallery, such as Room 8, Assyria: Nimrud, Room 10, or Assyria: Lion hunts.",
            "A generic museum label, unreadable sign, exhibit caption, audio-guide number, or object label is not direct_room_label.",
            "Use score 0 for visible contradiction, 2 for barely plausible weak_generic evidence, 4 for plausible generic objects or layout, 6 for several visible entities that match but are not unique, 8 for strong shared_theme evidence or likely room_specific evidence, 9 for very strong room_specific evidence, and 10 only for direct_room_label or unmistakable unique current-room evidence.",
            "A score of 10 is not allowed for shared_theme evidence.",
            "If multiple rooms share the same visible entities, give them similar scores and do not force a single winner.",
            "If the image lacks unique landmarks or legible direct room labels, the top candidate should not be much higher than similar nearby rooms.",
            "A broad Assyrian theme match should not by itself distinguish Room 10 from Rooms 7, 8, and 9.",
            "Keep each room score reason to at most twelve words.",
            "Return at most six short evidence_entities and keep the summary under forty words.",
            "Do not invent room ids outside the candidate list.",
            "Return JSON only.",
        ]
    )


def build_visual_detection_localization_input(
    *,
    captures: list[dict[str, object]],
    candidates: list[dict],
) -> str:
    view_labels = []
    for capture in captures:
        label = capture.get("label")
        heading = capture.get("heading")
        if not isinstance(label, str) or not label:
            continue
        if isinstance(heading, (int, float)):
            view_labels.append(f"{label} ({float(heading):.1f} deg)")
        else:
            view_labels.append(label)

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
        _append_visual_profile_parts(parts, candidate)
        candidate_lines.append("- " + " | ".join(parts))

    lines = [
        f"These are {len(captures)} overlapping views from the same panorama.",
        "Aggregate all visible evidence across the full panorama before answering.",
    ]
    if view_labels:
        lines.append("Available views: " + ", ".join(view_labels) + ".")
    lines.extend(
        [
            "",
            f"Candidate room count: {len(candidates)}.",
            "Candidate rooms:",
            *candidate_lines,
            "",
            "Task:",
            "1. Identify at most eight distinct museum-relevant entities across all views, merging repeated sightings.",
            "2. Assign location_scope=inside for entities in the current gallery, outside for entities visible only through adjacent openings, and unknown when uncertain.",
            "3. For every candidate room, classify evidence_type before scoring.",
            "4. Use inside entities and current-gallery visual themes to assign an independent evidence score from 0 to 10 for every candidate room.",
            "5. Output exactly one room_scores entry for each candidate room with room_id, evidence_type, score, and reason. These scores are raw evidence, not normalized probabilities.",
            "6. Preserve passages and outside entities in the entity list even when they are not used as primary localization evidence.",
        ]
    )
    return "\n".join(lines)


def build_visual_detection_localization_schema(
    room_ids: list[str],
    *,
    include_bounds: bool = True,
) -> dict:
    entity_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                **({"maxLength": 120} if include_bounds else {}),
            },
            "kind": {"type": "string", "enum": list(VIEW_DETECTION_KINDS)},
            "confidence": {"type": "number"},
            "source_views": {
                "type": "array",
                "items": {"type": "string"},
                **({"maxItems": 8} if include_bounds else {}),
            },
            "location_scope": {"type": "string", "enum": list(ENTITY_LOCATION_SCOPES)},
        },
        "required": ["name", "kind", "confidence", "source_views", "location_scope"],
        "additionalProperties": False,
    }
    room_score_schema = {
        "type": "object",
        "properties": {
            "room_id": {"type": "string", "enum": room_ids},
            "evidence_type": {"type": "string", "enum": list(ROOM_EVIDENCE_TYPES)},
            "score": {"type": "number"},
            "reason": {
                "type": "string",
                **({"maxLength": 160} if include_bounds else {}),
            },
        },
        "required": ["room_id", "evidence_type", "score", "reason"],
        "additionalProperties": False,
    }
    visual_localization_schema = {
        "type": "object",
        "properties": {
            "predicted_room_id": {"type": ["string", "null"], "enum": room_ids + [None]},
            "room_scores": {
                "type": "array",
                "items": room_score_schema,
                **(
                    {"minItems": len(room_ids), "maxItems": len(room_ids)}
                    if include_bounds
                    else {}
                ),
            },
            "evidence_entities": {
                "type": "array",
                "items": {
                    "type": "string",
                    **({"maxLength": 120} if include_bounds else {}),
                },
                **({"maxItems": 6} if include_bounds else {}),
            },
            "summary": {
                "type": "string",
                **({"maxLength": 240} if include_bounds else {}),
            },
        },
        "required": ["predicted_room_id", "room_scores", "evidence_entities", "summary"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": entity_schema,
                **({"maxItems": 8} if include_bounds else {}),
            },
            "visual_localization": visual_localization_schema,
        },
        "required": ["entities", "visual_localization"],
        "additionalProperties": False,
    }


def build_visual_theme_direction_instructions() -> str:
    return " ".join(
        [
            "You select one visual direction for museum indoor navigation.",
            "You are given eight or fewer overlapping panorama views and only the visual theme of the next subgoal gallery.",
            "Choose the single view whose visible scene most plausibly leads toward or reveals that theme.",
            "Use visible exhibits, architecture, signs, doorways, and adjacent-gallery glimpses.",
            "Do not infer a room id, map direction, compass direction, heading, or spatial alignment.",
            "If no view clearly matches, still choose the most plausible view and lower confidence.",
            "Return exactly one selected_view_id and one short visual_evidence sentence.",
            "Return JSON only.",
        ]
    )


def build_visual_theme_direction_input(*, view_ids: list[str], subgoal_theme: dict) -> str:
    visual_cues = [
        value
        for value in subgoal_theme.get("visual_cues", [])
        if isinstance(value, str) and value
    ][:4]
    lines = [
        "Select one panorama view as the next movement direction.",
        "The views are ordered clockwise, but no compass headings or map directions are provided.",
        f"Available view ids: {', '.join(view_ids)}.",
        "",
        "Subgoal visual theme:",
        f"- theme_name: {subgoal_theme.get('title') or 'unknown'}",
        f"- category: {subgoal_theme.get('category') or 'unknown'}",
        f"- short_description: {subgoal_theme.get('short_description') or 'unknown'}",
    ]
    if visual_cues:
        lines.append("- visual_cues: " + "; ".join(visual_cues))
    lines.extend(
        [
            "",
            "Judge only from the supplied images and this visual theme.",
            "Return exactly one of the available view ids.",
        ]
    )
    return "\n".join(lines)


def build_visual_theme_direction_schema(view_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "selected_view_id": {"type": "string", "enum": view_ids},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "visual_evidence": {"type": "string", "maxLength": 240},
        },
        "required": ["selected_view_id", "confidence", "visual_evidence"],
        "additionalProperties": False,
    }


def build_navigation_parse_instructions() -> str:
    return " ".join(
        [
            "You are a museum navigation query parser.",
            "Extract only ordered waypoint entities and ordered goal entities from the user query.",
            "Classify each extracted entity as gallery or artwork.",
            "Do not extract or return source entities, source rooms, current locations, or start locations.",
            "Treat explicit room, gallery, stairs, or lift mentions as gallery entities.",
            "Also treat a gallery described by its collection, culture, period, or visual theme as a gallery entity.",
            "Treat named objects, sculptures, monuments, reliefs, gates, statues, and artworks as artwork entities.",
            "Treat entities mentioned after from, starting at, current, or currently in as source context and ignore them.",
            "Treat entities mentioned after passing, via, through, or by way of as waypoints.",
            "Treat entities mentioned after to, toward, visit, find, go to, navigate to, or take me to as goals.",
            "Preserve entity order from the original query.",
            "Infer gallery and artwork room ids using only the provided gallery themes.",
            "A thematic gallery description may resolve to any allowed room whose theme is the best semantic match.",
            "Do not invent room ids, entities, source fields, or extra steps.",
            "For an explicit Room ID, predicted_room_id must match that ID and confidence must be 1.0.",
            "For a theme-described gallery or artwork, predicted_room_id must be one allowed room id and confidence must be between 0 and 1.",
            "Return JSON only.",
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
            "1. Identify waypoint and goal spans from the wording.",
            "2. Ignore source/current/start spans.",
            "3. Classify each extracted entity as gallery or artwork; thematic collection descriptions are galleries.",
            "4. Ground each entity to one allowed room id using the gallery themes.",
            "5. Return no source fields.",
            "",
            f"Query: {instruction}",
        ]
    )


def build_navigation_parse_schema(room_ids: list[str]) -> dict:
    entity_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 120},
            "entity_type": {"type": "string", "enum": ["gallery", "artwork"]},
            "predicted_room_id": {"type": ["string", "null"], "enum": room_ids + [None]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["name", "entity_type", "predicted_room_id", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "goal_entities": {"type": "array", "items": entity_schema},
            "waypoint_entities": {"type": "array", "items": entity_schema},
        },
        "required": ["goal_entities", "waypoint_entities"],
        "additionalProperties": False,
    }


def build_spatial_context_extraction_instructions() -> str:
    return " ".join(
        [
            "You are a museum panorama sector analyst.",
            "Inspect each panorama sector independently while remembering the sectors are clockwise views from one location.",
            "The panorama heading is unknown; do not assume view_0 is north, front, right, rear, or left.",
            "Do not decide the current room in this step.",
            "Use the provided theme vocabulary only as labels for visible evidence, not as priors for localization.",
            "List visible gallery themes, exhibit themes, signs, thresholds, openings, and adjacent-room glimpses.",
            "A single sector may legitimately show multiple adjacent gallery themes at once.",
            "If a sector is mostly blank wall, ceiling, floor, glare, cropped signage, or lacks identifiable gallery evidence, return an empty themes list and state that no reliable theme is visible.",
            "Do not guess a room theme from weak, cropped, or partial evidence.",
            "Only assign a theme when visible objects, signs, displays, or architectural cues support it.",
            "If the exact exhibit identity is uncertain but a broad gallery theme is visibly supported, return the broad theme with lower confidence; otherwise return no theme.",
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
            "3. Keep the answer grounded in what is visible in that sector; do not localize the camera.",
            "4. If a sector contains views into adjacent galleries, include those adjacent gallery themes instead of suppressing them.",
            "5. If a sector has no reliable gallery evidence, return themes=[] for that sector.",
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
            "themes": {
                "type": "array",
                "items": theme_schema,
                "maxItems": 4,
            },
            "summary": {"type": "string"},
        },
        "required": ["view_id", "themes", "summary"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "views": {
                "type": "array",
                "items": view_schema,
                "minItems": len(view_ids),
                "maxItems": len(view_ids),
            },
            "summary": {"type": "string"},
        },
        "required": ["views", "summary"],
        "additionalProperties": False,
    }


def build_spatial_alignment_instructions(*, direct_images: bool = False) -> str:
    base_parts = [
        "You are a museum spatial alignment refiner.",
        "Choose which candidate room best explains the clockwise panorama-sector observations after allowing one global rotation.",
        "The sector labels are not compass directions.",
        "Use candidate room profiles and neighbor relations to test spatial consistency.",
        "Separate evidence inside the current room from evidence visible through openings into adjacent rooms.",
        "Treat sectors with no reliable theme as neutral, not as support for any room.",
        "Do not choose a room only because one visible theme matches; prefer the candidate whose room profile and neighboring-room layout jointly explain the sector pattern.",
        "Infer which allocentric direction the first panorama sector view_0 most likely corresponds to.",
        "Use only the provided candidate rooms.",
        "Keep the JSON compact: use at most three evidence strings, keep each evidence string under twenty words, and keep the summary under twenty words.",
    ]
    if direct_images:
        base_parts.extend(
            [
                "When panorama images are provided directly, do not jump straight to a room prediction from one sign or one iconic object.",
                "First infer the dominant visible theme of every panorama sector.",
                "The supplied numeric headings are geocentric bearings in one local frame: zero is an unknown frame origin and angles increase clockwise.",
                "Do not treat a geocentric heading of 0 degrees as map north.",
                "There is one constant global rotation from the geocentric frame to the allocentric room-map frame.",
                "Explicitly test the four rotations obtained by assigning view_0 to north, east, south, or west.",
                "Use the heading differences between views to propagate each tested rotation to every sector.",
                "Treat every candidate room as a complete spatial hypothesis: that candidate is the current room and its listed directional neighbors explain adjacent-room views.",
                "Evaluate all candidate hypotheses before choosing predicted_room_id; do not choose a current room first and then retrofit ambiguous adjacent views to its neighbors.",
                "For each hypothesis, compare current-room cues, directionally aligned neighbor cues, and contradictions.",
                "An exact room title, exhibit type, or specific visual-cue match outweighs a broad category match.",
                "Do not map an adjacent view to a neighboring room from category alone when another hypothesis has a directionally consistent neighbor with a more specific visual match.",
                "Assign each adjacent-room view one best room_hint from the supplied profiles, or null when genuinely uncertain; do not use an A-or-B room identity to create support.",
                "Treat a specific non-null room_hint as fixed evidence when evaluating candidate hypotheses.",
                "A candidate's neighbor_support may use only a room explicitly listed as that candidate's neighbor at the aligned allocentric direction.",
                "If a view's best room_hint is not that candidate's listed neighbor at the aligned direction, record it as a contradiction instead of neighbor support.",
                "Do not replace a specific room_hint with a visually broader same-category neighbor merely to preserve a preferred current-room hypothesis.",
                "For every candidate hypothesis, explicitly inspect the panorama sectors aligned with every listed neighbor direction, even when no adjacent room was noticed initially.",
                "Classify each directional neighbor check as specific_match, broad_match, different_room, not_visible, or uncertain.",
                "A not_visible or occluded neighbor is neutral rather than a contradiction.",
                "Multiple specific neighbor matches in different directions are stronger evidence than one iconic current-room object.",
                "Then align at least two sectors to allocentric directions and the most specific candidate-room or neighboring-room themes before deciding the room.",
                "Your evidence and alignment trace must explicitly reference sector ids such as view_2 or view_5.",
                "Return one view_observations entry for every supplied sector, one candidate_hypotheses entry for every candidate, and rank every candidate room exactly once.",
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
    view_headings: list[dict[str, object]] | None = None,
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
        "3. Separate current-room evidence from adjacent-room or visible-only evidence before ranking.",
        "4. Output room_ranking as an ordered top-k list by spatial consistency.",
        "5. Scores are relative support only; do not normalize them and do not treat them as probabilities.",
        "6. Briefly summarize the main supporting and conflicting evidence.",
    ]
    if len(view_ids) == 4:
        lines.insert(
            6,
            "These four sectors are spaced exactly 90 degrees apart; their labels are view ids, not map compass directions.",
        )
    if direct_images:
        headings_by_view = {
            str(record.get("view_id")): float(record["heading_deg"])
            for record in (view_headings or [])
            if isinstance(record, dict)
            and isinstance(record.get("view_id"), str)
            and isinstance(record.get("heading_deg"), (int, float))
        }
        if headings_by_view:
            lines.extend(
                [
                    "",
                    "Geocentric sector headings (degrees clockwise in an unknown local frame):",
                    *[
                        f"- {view_id}: {headings_by_view[view_id]:.6f} degrees"
                        for view_id in view_ids
                        if view_id in headings_by_view
                    ],
                    "Use one constant rotation for all sectors; do not interpret these values as allocentric compass headings directly.",
                ]
            )
        lines.extend(
            [
                "7. Produce one view_observations record for every sector before choosing the room.",
                "8. For every candidate, build a joint hypothesis in which it is the current room and each adjacent-room view must match one of its listed neighbors at the aligned allocentric direction.",
                "9. Compare hypotheses before selecting the room. Record current-room support, neighbor support, and contradictions for every candidate; category-only similarity is weak support.",
                "9a. Choose one best room_hint for each adjacent-room view before scoring candidate hypotheses; use null if the image cannot distinguish the profiles.",
                "9b. A neighbor_support statement is valid only when that fixed room_hint is a listed neighbor of the hypothesis room at the aligned direction.",
                "9c. Otherwise put the view in contradictions. Never count a non-neighbor or an A-or-B identity as neighbor support.",
                "9d. For each hypothesis, output one directional_neighbor_checks item for every listed neighbor, inspecting the view ids nearest that neighbor's allocentric direction.",
                "9e. Each directional check must contain allocentric_direction, expected_neighbor_room_id, inspected_view_ids, status, observed_room_hint, and rationale.",
                "9f. Use status specific_match, broad_match, different_room, not_visible, or uncertain. Do not penalize not_visible; reward multiple directionally independent specific_match results.",
                "10. Test view_0 as north, east, south, and west, propagating each rotation using the numeric heading differences.",
                "11. Output a sector_alignment trace that maps at least two panorama sectors to allocentric directions and the most specifically matching candidate-room theme or neighbor.",
                "12. Do not rely only on one room sign or iconic object if the wider spatial pattern points elsewhere.",
                "13. Return these top-level JSON fields: predicted_room_id, confidence, view_0_allocentric_direction, evidence, room_ranking, summary, sector_alignment, view_observations, and candidate_hypotheses.",
                "14. Each room_ranking item must contain room_id and numeric score, with every candidate room appearing exactly once.",
                "15. Each candidate_hypotheses item must contain room_id, numeric score, current_room_support, neighbor_support, and contradictions, with every candidate appearing exactly once.",
                "16. current_room_support, neighbor_support, and contradictions must be arrays of strings and must name relevant view ids when visual evidence is used.",
                "16a. Each candidate_hypotheses item must also contain directional_neighbor_checks.",
                "16b. directional_neighbor_checks must cover every neighbor listed for that candidate exactly once.",
                "17. Each view_observations item must contain view_id, themes, location_scope, room_hint, and summary; location_scope must be current_room, adjacent_room, mixed, or uncertain.",
                "18. Each sector_alignment item must contain view_id, allocentric_direction, matched_room_id, matched_theme, and rationale.",
                "19. matched_room_id and room_hint may name a candidate room, one of its listed neighboring rooms, or null.",
                "20. confidence must be numeric, evidence and themes must be arrays of strings, and summary fields must be strings.",
            ]
        )
    return "\n".join(lines)


def build_spatial_alignment_schema(
    room_ids: list[str],
    *,
    view_ids: list[str] | None = None,
    include_sector_alignment: bool = False,
    include_view_observations: bool = False,
    include_candidate_hypotheses: bool = False,
    context_room_ids: list[str] | None = None,
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
            "predicted_room_id": {"type": "string", "enum": room_ids},
            "confidence": {"type": "number"},
            "view_0_allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
            "room_ranking": {
                "type": "array",
                "items": room_score_schema,
                "minItems": len(room_ids),
                "maxItems": len(room_ids),
            },
            "summary": {"type": "string"},
        },
        "required": [
            "predicted_room_id",
            "confidence",
            "view_0_allocentric_direction",
            "evidence",
            "room_ranking",
            "summary",
        ],
        "additionalProperties": False,
    }
    if include_sector_alignment:
        matched_room_ids = list(context_room_ids or room_ids)
        sector_schema = {
            "type": "object",
            "properties": {
                "view_id": {"type": "string", "enum": list(view_ids or [])},
                "allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
                "matched_room_id": {"type": ["string", "null"], "enum": matched_room_ids + [None]},
                "matched_theme": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["view_id", "allocentric_direction", "matched_room_id", "matched_theme", "rationale"],
            "additionalProperties": False,
        }
        schema["properties"]["sector_alignment"] = {
            "type": "array",
            "items": sector_schema,
            "minItems": 2,
            "maxItems": len(view_ids or []),
        }
        schema["required"].append("sector_alignment")
    if include_view_observations:
        visible_room_ids = list(context_room_ids or room_ids)
        view_observation_schema = {
            "type": "object",
            "properties": {
                "view_id": {"type": "string", "enum": list(view_ids or [])},
                "themes": {"type": "array", "items": {"type": "string"}},
                "location_scope": {
                    "type": "string",
                    "enum": ["current_room", "adjacent_room", "mixed", "uncertain"],
                },
                "room_hint": {"type": ["string", "null"], "enum": visible_room_ids + [None]},
                "summary": {"type": "string"},
            },
            "required": ["view_id", "themes", "location_scope", "room_hint", "summary"],
            "additionalProperties": False,
        }
        schema["properties"]["view_observations"] = {
            "type": "array",
            "items": view_observation_schema,
            "minItems": len(view_ids or []),
            "maxItems": len(view_ids or []),
        }
        schema["required"].append("view_observations")
        schema["properties"]["room_ranking"]["minItems"] = len(room_ids)
        schema["properties"]["room_ranking"]["maxItems"] = len(room_ids)
    if include_candidate_hypotheses:
        hypothesis_schema = {
            "type": "object",
            "properties": {
                "room_id": {"type": "string", "enum": room_ids},
                "score": {"type": "number"},
                "current_room_support": {"type": "array", "items": {"type": "string"}},
                "neighbor_support": {"type": "array", "items": {"type": "string"}},
                "contradictions": {"type": "array", "items": {"type": "string"}},
                "directional_neighbor_checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "allocentric_direction": {
                                "type": "string",
                                "enum": list(ALLOCENTRIC_DIRECTIONS),
                            },
                            "expected_neighbor_room_id": {
                                "type": "string",
                                "enum": list(context_room_ids or room_ids),
                            },
                            "inspected_view_ids": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(view_ids or []),
                                },
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "specific_match",
                                    "broad_match",
                                    "different_room",
                                    "not_visible",
                                    "uncertain",
                                ],
                            },
                            "observed_room_hint": {
                                "type": ["string", "null"],
                                "enum": list(context_room_ids or room_ids) + [None],
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "allocentric_direction",
                            "expected_neighbor_room_id",
                            "inspected_view_ids",
                            "status",
                            "observed_room_hint",
                            "rationale",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "room_id",
                "score",
                "current_room_support",
                "neighbor_support",
                "contradictions",
                "directional_neighbor_checks",
            ],
            "additionalProperties": False,
        }
        schema["properties"]["candidate_hypotheses"] = {
            "type": "array",
            "items": hypothesis_schema,
            "minItems": len(room_ids),
            "maxItems": len(room_ids),
        }
        schema["required"].append("candidate_hypotheses")
    return schema
