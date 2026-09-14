from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from st_nav import Observation, RenderedView
from st_nav.experiments.map_grounded_alignment import (
    MapGroundedSpatialAlignmentExperiment,
    _relax_json_schema,
    build_candidate_cards,
    build_map_grounded_decision_schema,
    recover_map_grounded_top1_prefix,
    render_semantic_map,
    normalize_map_grounded_output,
    validate_map_grounded_decision_output,
    validate_map_grounded_compact_reasoning_output,
    validate_map_grounded_output,
    validate_map_grounded_top1_output,
    validate_map_regions,
)


CANDIDATES = ["Room 7", "Room 8", "Room 9", "Room 10", "Room 23"]
HEADINGS = [330.0, 15.0, 60.0, 105.0, 150.0, 195.0, 240.0, 285.0]


def mini_room_graph() -> dict[str, dict]:
    records = {
        "Room 7": ("Assyria: Nimrud", "Middle East", "Room 8", "north"),
        "Room 8": ("Assyria: Nimrud", "Middle East", "Room 7", "south"),
        "Room 9": ("Assyria: Nineveh", "Middle East", "Room 8", "south"),
        "Room 10": ("Assyria: Lion hunts", "Middle East", "Room 23", "north"),
        "Room 23": ("Greek and Roman sculpture", "Ancient Greece and Rome", "Room 10", "south"),
    }
    graph = {}
    for room_id, (title, category, neighbor_id, direction) in records.items():
        graph[room_id] = {
            "room_id": room_id,
            "floor": "0",
            "title": title,
            "category": category,
            "visual_profile": {
                "short_description": f"Description for {title}.",
                "visual_cues": ["shared relief panels", f"specific cue for {room_id}"],
            },
            "neighbors": [
                {
                    "target_room_id": neighbor_id,
                    "allocentric_direction": direction,
                    "transition_type": "passage",
                }
            ],
        }
    return graph


def mini_regions() -> dict:
    rooms = {}
    for index, room_id in enumerate(sorted(CANDIDATES)):
        left = 0.05 + index * 0.18
        rooms[room_id] = {
            "polygons": [[[left, 0.2], [left + 0.12, 0.2], [left + 0.12, 0.7], [left, 0.7]]],
            "label_anchor": [left + 0.06, 0.45],
        }
    return {
        "coordinate_system": "normalized_0_1",
        "north_up": True,
        "rooms": rooms,
    }


def build_observation(directory: Path) -> Observation:
    views = []
    for index, heading in enumerate(HEADINGS):
        path = directory / f"view_{index}.png"
        Image.new("RGB", (24, 24), (index * 20, 80, 120)).save(path)
        views.append(RenderedView(label=f"secret_cardinal_label_{index}", heading=heading, path=str(path)))
    return Observation(
        pano_id="pano-test",
        views=views,
        metadata={"ground_truth_room_id": "GroundTruthOnly", "visual_localization_score": 9.99},
    )


def rotation_tests(best_score: float) -> list[dict]:
    if best_score == 9:
        components = (3.0, 3.0, 3.0)
    elif best_score == 8:
        components = (3.0, 3.0, 2.0)
    elif best_score == 7:
        components = (3.0, 2.0, 2.0)
    elif best_score == 6:
        components = (2.0, 2.0, 2.0)
    else:
        components = (2.0, 2.0, 1.0)
    records = []
    for direction in ("north", "east", "south", "west"):
        current, neighbor, layout = components if direction == "north" else (1.0, 1.0, 1.0)
        records.append(
            {
                "view_0_allocentric_direction": direction,
                "current_room_semantics": current,
                "neighbor_semantics": neighbor,
                "directional_layout": layout,
                "contradiction_penalty": 0.0,
                "total_score": current + neighbor + layout,
                "supporting_view_ids": ["view_0", "view_2"],
                "contradictions": [],
                "summary": f"{direction} rotation summary",
            }
        )
    return records


def valid_output(graph: dict[str, dict]) -> dict:
    best_scores = {"Room 8": 9.0, "Room 7": 8.0, "Room 9": 7.0, "Room 10": 6.0, "Room 23": 5.0}
    hypotheses = []
    for room_id in sorted(CANDIDATES):
        neighbor = graph[room_id]["neighbors"][0]
        hypotheses.append(
            {
                "room_id": room_id,
                "rotation_tests": rotation_tests(best_scores[room_id]),
                "selected_view_0_allocentric_direction": "north",
                "directional_neighbor_checks": [
                    {
                        "expected_neighbor_room_id": neighbor["target_room_id"],
                        "allocentric_direction": neighbor["allocentric_direction"],
                        "inspected_view_ids": ["view_2"],
                        "status": "specific_match",
                        "observed_room_hint": neighbor["target_room_id"],
                        "evidence_specificity": "specific",
                        "rationale": "The fixed view hint matches this neighbor.",
                    }
                ],
                "current_room_support": ["view_0 supports the room"],
                "neighbor_support": ["view_2 supports a listed neighbor"],
                "contradictions": [],
                "summary": f"Hypothesis for {room_id}",
            }
        )
    return {
        "predicted_room_id": "Room 8",
        "confidence": 0.82,
        "ambiguity_level": "low",
        "view_0_allocentric_direction": "north",
        "geocentric_to_allocentric_offset_deg": 30.0,
        "view_observations": [
            {
                "view_id": f"view_{index}",
                "themes": ["museum gallery"],
                "location_scope": "current_room" if index != 2 else "adjacent_room",
                "room_hints": [
                    {
                        "room_id": "Room 8" if index != 2 else "Room 7",
                        "specificity": "specific",
                        "evidence": "Fixed evidence before hypothesis testing.",
                    }
                ],
                "opening_or_transition_cues": [],
                "summary": "Independent view observation.",
            }
            for index in range(8)
        ],
        "candidate_hypotheses": hypotheses,
        "room_ranking": [
            {"room_id": "Room 8", "score": 9.0},
            {"room_id": "Room 7", "score": 8.0},
            {"room_id": "Room 9", "score": 7.0},
            {"room_id": "Room 10", "score": 6.0},
            {"room_id": "Room 23", "score": 5.0},
        ],
        "top1_vs_top2": {
            "top1_room_id": "Room 8",
            "top2_room_id": "Room 7",
            "margin": 1.0,
            "decisive_discriminator": "Room 8 has the directionally consistent Room 7 neighbor.",
            "comparison_summary": "The candidates share a theme, so topology decides the room.",
        },
        "map_alignment_trace": [
            {
                "view_id": "view_0",
                "allocentric_direction": "north",
                "mapped_room_id": "Room 8",
                "evidence_type": "current_room",
                "evidence_specificity": "specific",
                "rationale": "view_0 aligns north under the selected rotation.",
            },
            {
                "view_id": "view_2",
                "allocentric_direction": "east",
                "mapped_room_id": "Room 7",
                "evidence_type": "neighbor",
                "evidence_specificity": "specific",
                "rationale": "view_2 aligns east under the same rotation.",
            },
        ],
        "summary": "Room 8 is the strongest map-grounded hypothesis.",
    }


def compact_reasoning_output() -> dict:
    best_scores = {"Room 8": 9.0, "Room 7": 8.0, "Room 9": 7.0, "Room 10": 6.0, "Room 23": 5.0}
    return {
        "predicted_room_id": "Room 8",
        "confidence": 0.82,
        "view_0_allocentric_direction": "north",
        "candidate_rotation_scores": [
            {
                "room_id": room_id,
                "north": score,
                "east": score - 1.0,
                "south": score - 2.0,
                "west": score - 1.0,
            }
            for room_id, score in sorted(best_scores.items())
        ],
        "top2_room_id": "Room 7",
        "supporting_views": ["view_0", "view_2"],
        "matched_neighbors": [{"direction": "south", "room_id": "Room 7", "view_id": "view_2"}],
        "decisive_discriminator": "Room 8 has the strongest room evidence and a south Room 7 transition.",
    }


class MapGroundedSpatialAlignmentTests(unittest.TestCase):
    def test_gemini_schema_relaxation_preserves_contract_constraints(self) -> None:
        schema = {
            "type": "array",
            "items": {"type": ["string", "null"], "enum": ["north", "east", None]},
            "minItems": 2,
            "maxItems": 2,
            "additionalProperties": False,
        }
        relaxed = _relax_json_schema(schema)
        self.assertEqual(relaxed["items"]["type"], ["string", "null"])
        self.assertNotIn("enum", relaxed["items"])
        self.assertNotIn("minItems", relaxed)
        self.assertNotIn("maxItems", relaxed)
        self.assertNotIn("additionalProperties", relaxed)

    def test_output_normalization_repairs_only_conservative_diagnostic_fields(self) -> None:
        graph = mini_room_graph()
        payload = valid_output(graph)
        payload["ambiguity_level"] = "none"
        payload["view_observations"][0]["location_scope"] = "mixed"
        payload["view_observations"][0]["room_hints"][0]["specificity"] = "room_specific"
        payload["view_observations"][0]["room_hints"][0]["evidence"] = ["cue"]
        payload["view_observations"].pop()
        winner = next(item for item in payload["candidate_hypotheses"] if item["room_id"] == "Room 8")
        winner["directional_neighbor_checks"] = []
        views = [{"view_id": f"view_{index}", "heading": heading} for index, heading in enumerate(HEADINGS)]
        normalized, changes = normalize_map_grounded_output(
            payload,
            candidate_room_ids=CANDIDATES,
            room_graph=graph,
            views=views,
        )
        self.assertEqual(normalized["predicted_room_id"], "Room 8")
        self.assertEqual(normalized["room_ranking"], payload["room_ranking"])
        self.assertEqual(normalized["ambiguity_level"], "high")
        self.assertEqual(normalized["view_observations"][0]["location_scope"], "uncertain")
        self.assertEqual(normalized["view_observations"][0]["room_hints"][0]["specificity"], "broad")
        self.assertEqual(len(normalized["view_observations"]), 8)
        normalized_winner = next(
            item for item in normalized["candidate_hypotheses"] if item["room_id"] == "Room 8"
        )
        self.assertEqual(len(normalized_winner["directional_neighbor_checks"]), 1)
        self.assertEqual(normalized_winner["directional_neighbor_checks"][0]["status"], "uncertain")
        self.assertTrue(changes)
        validate_map_grounded_output(
            normalized,
            candidate_room_ids=CANDIDATES,
            room_graph=graph,
            views=views,
        )

    def test_output_normalization_sanitizes_support_and_syncs_winning_rotation(self) -> None:
        graph = mini_room_graph()
        payload = valid_output(graph)
        winner = next(
            hypothesis
            for hypothesis in payload["candidate_hypotheses"]
            if hypothesis["room_id"] == payload["predicted_room_id"]
        )
        east = next(
            rotation
            for rotation in winner["rotation_tests"]
            if rotation["view_0_allocentric_direction"] == "east"
        )
        east.update(
            {
                "current_room_semantics": 3.0,
                "neighbor_semantics": 3.0,
                "directional_layout": 4.0,
                "contradiction_penalty": 0.0,
                "total_score": 10.0,
                "supporting_view_ids": ["view_0", "invalid-view", "view_0"],
            }
        )
        payload["room_ranking"][0]["score"] = 10.0
        payload["top1_vs_top2"]["margin"] = 2.0
        views = [{"view_id": f"view_{index}", "heading": heading} for index, heading in enumerate(HEADINGS)]

        normalized, changes = normalize_map_grounded_output(
            payload,
            candidate_room_ids=CANDIDATES,
            room_graph=graph,
            views=views,
        )

        normalized_winner = next(
            hypothesis
            for hypothesis in normalized["candidate_hypotheses"]
            if hypothesis["room_id"] == normalized["predicted_room_id"]
        )
        normalized_east = next(
            rotation
            for rotation in normalized_winner["rotation_tests"]
            if rotation["view_0_allocentric_direction"] == "east"
        )
        self.assertEqual(normalized_east["supporting_view_ids"], ["view_0"])
        self.assertEqual(normalized_winner["selected_view_0_allocentric_direction"], "east")
        self.assertEqual(normalized["view_0_allocentric_direction"], "east")
        self.assertEqual(normalized["geocentric_to_allocentric_offset_deg"], 120.0)
        self.assertIn("Room 8.selected_rotation->highest_score", changes)
        validate_map_grounded_output(
            normalized,
            candidate_room_ids=CANDIDATES,
            room_graph=graph,
            views=views,
        )

    def test_real_region_catalog_covers_all_level_zero_graph_nodes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        graph = json.loads(
            (root / "dataset/sites/british_museum/normalized/room_graph_with_visual_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        regions = json.loads(
            (root / "dataset/sites/british_museum/maps/level_0_room_regions.json").read_text(encoding="utf-8")
        )
        validate_map_regions(graph, regions)

    def test_semantic_map_uses_lexical_candidate_labels_and_renders_sidebar(self) -> None:
        graph = mini_room_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            base_map = directory / "base.png"
            output = directory / "semantic.png"
            Image.new("RGBA", (400, 300), (230, 230, 230, 255)).save(base_map)
            diagnostics = render_semantic_map(
                base_map_path=base_map,
                region_payload=mini_regions(),
                room_graph=graph,
                candidate_room_ids=["Room 9", "Room 8", "Room 23", "Room 7", "Room 10"],
                output_path=output,
            )
            with Image.open(output) as rendered_image:
                rendered_size = rendered_image.size
        self.assertEqual(
            diagnostics["candidate_map_labels"],
            {"Room 10": "C1", "Room 23": "C2", "Room 7": "C3", "Room 8": "C4", "Room 9": "C5"},
        )
        self.assertEqual(rendered_size, (900, 300))
        self.assertTrue(diagnostics["north_up"])

    def test_candidate_cards_mark_shared_theme_without_rank_or_score(self) -> None:
        cards = build_candidate_cards(mini_room_graph(), CANDIDATES)
        room7 = next(card for card in cards if card["room_id"] == "Room 7")
        self.assertEqual(room7["same_title_candidates"], ["Room 8"])
        self.assertIn("Room 8", room7["same_category_candidates"])
        self.assertIn("shared relief panels", room7["shared_non_decisive_visual_cues"])
        serialized = json.dumps(cards)
        self.assertNotIn('"rank"', serialized)
        self.assertNotIn('"score"', serialized)
        self.assertNotIn('"reason"', serialized)

    def test_request_contains_map_first_and_exactly_nine_images_without_leakage(self) -> None:
        graph = mini_room_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=lambda _: {"output_text": "{}"},
            )
            request, cards, _ = experiment.build_request_body(
                observation=observation,
                manifest={"heading_mode": "museum", "captures": [{"label": "secret-north"}]},
                candidate_room_ids=CANDIDATES,
                semantic_map_path=semantic_map,
            )
        content = request["input"][0]["content"]
        image_blocks = [record for record in content if record.get("type") == "input_image"]
        text = "\n".join(str(record.get("text", "")) for record in content if record.get("type") == "input_text")
        self.assertEqual(len(image_blocks), 9)
        self.assertEqual(request["max_output_tokens"], 1024)
        schema = request["text"]["format"]["schema"]
        self.assertEqual(
            schema["required"],
            [
                "predicted_room_id",
                "confidence",
                "view_0_allocentric_direction",
                "candidate_rotation_scores",
                "top2_room_id",
                "supporting_views",
                "matched_neighbors",
                "decisive_discriminator",
            ],
        )
        self.assertEqual(
            set(schema["properties"]),
            {
                "predicted_room_id",
                "confidence",
                "view_0_allocentric_direction",
                "candidate_rotation_scores",
                "top2_room_id",
                "supporting_views",
                "matched_neighbors",
                "decisive_discriminator",
            },
        )
        self.assertNotIn("candidate_hypotheses", request["instructions"])
        self.assertNotIn("rotation_tests", request["instructions"])
        self.assertIn("map_reference", content[1]["text"])
        self.assertEqual(content[2]["type"], "input_image")
        self.assertIn("view_0", content[3]["text"])
        self.assertNotIn("GroundTruthOnly", text)
        self.assertNotIn("9.99", text)
        self.assertNotIn("secret_cardinal_label", text)
        self.assertNotIn("secret-north", text)
        self.assertEqual([card["map_label"] for card in cards], ["C1", "C2", "C3", "C4", "C5"])

    def test_validator_accepts_coherent_full_output_and_rejects_bad_score(self) -> None:
        graph = mini_room_graph()
        views = [{"view_id": f"view_{index}", "heading": heading} for index, heading in enumerate(HEADINGS)]
        payload = valid_output(graph)
        validate_map_grounded_output(payload, candidate_room_ids=CANDIDATES, room_graph=graph, views=views)
        winner = next(
            hypothesis
            for hypothesis in payload["candidate_hypotheses"]
            if hypothesis["room_id"] == payload["predicted_room_id"]
        )
        winner["rotation_tests"][0]["total_score"] += 0.5
        with self.assertRaisesRegex(ValueError, "does not match the rubric"):
            validate_map_grounded_output(payload, candidate_room_ids=CANDIDATES, room_graph=graph, views=views)

    def test_validator_treats_nonwinning_hypothesis_details_as_audit_only(self) -> None:
        graph = mini_room_graph()
        views = [{"view_id": f"view_{index}", "heading": heading} for index, heading in enumerate(HEADINGS)]
        payload = valid_output(graph)
        nonwinner = next(
            hypothesis for hypothesis in payload["candidate_hypotheses"] if hypothesis["room_id"] == "Room 7"
        )
        nonwinner["rotation_tests"] = [{"supporting_view_ids": ["invalid-view"]}]
        nonwinner["selected_view_0_allocentric_direction"] = "invalid-direction"
        nonwinner["directional_neighbor_checks"] = [{"inspected_view_ids": ["invalid-view"]}]
        validate_map_grounded_output(payload, candidate_room_ids=CANDIDATES, room_graph=graph, views=views)

        winner = next(
            hypothesis
            for hypothesis in payload["candidate_hypotheses"]
            if hypothesis["room_id"] == payload["predicted_room_id"]
        )
        winner["rotation_tests"] = [{"supporting_view_ids": ["invalid-view"]}]
        with self.assertRaisesRegex(ValueError, "exactly four rotation tests"):
            validate_map_grounded_output(payload, candidate_room_ids=CANDIDATES, room_graph=graph, views=views)

    def test_validator_accepts_partial_hypothesis_audit_when_winner_and_full_ranking_are_present(self) -> None:
        graph = mini_room_graph()
        views = [{"view_id": f"view_{index}", "heading": heading} for index, heading in enumerate(HEADINGS)]
        payload = valid_output(graph)
        payload["candidate_hypotheses"] = [
            hypothesis
            for hypothesis in payload["candidate_hypotheses"]
            if hypothesis["room_id"] == payload["predicted_room_id"]
        ]
        validate_map_grounded_output(payload, candidate_room_ids=CANDIDATES, room_graph=graph, views=views)
        payload = valid_output(graph)
        payload["candidate_hypotheses"] = [
            hypothesis for hypothesis in payload["candidate_hypotheses"] if hypothesis["room_id"] == "Room 7"
        ]
        with self.assertRaisesRegex(ValueError, "winning candidate"):
            validate_map_grounded_output(payload, candidate_room_ids=CANDIDATES, room_graph=graph, views=views)
        validate_map_grounded_decision_output(payload, candidate_room_ids=CANDIDATES)

    def test_run_once_can_score_decision_when_winning_audit_is_missing(self) -> None:
        graph = mini_room_graph()
        payload = valid_output(graph)
        payload["candidate_hypotheses"] = [
            hypothesis for hypothesis in payload["candidate_hypotheses"] if hypothesis["room_id"] == "Room 7"
        ]
        requests = []
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=lambda request: requests.append(request) or {"output_text": json.dumps(payload)},
                validation_retries=1,
                allow_incomplete_winning_hypothesis=True,
                compact_decision_output=False,
            )
            result = experiment.run_once(
                observation=observation,
                manifest={"heading_mode": "museum"},
                candidate_room_ids=CANDIDATES,
                semantic_map_path=semantic_map,
            )
        self.assertEqual(len(requests), 1)
        self.assertFalse(result["winning_hypothesis_audit_complete"])
        self.assertTrue(result["ranking_complete"])
        self.assertEqual(result["result"]["predicted_room_id"], "Room 8")
        self.assertEqual(result["audit_warnings"], ["candidate_hypotheses must include the winning candidate."])

    def test_top1_validation_ignores_malformed_ranking_but_locks_candidates(self) -> None:
        payload = valid_output(mini_room_graph())
        payload["room_ranking"][-1] = dict(payload["room_ranking"][1])
        validate_map_grounded_top1_output(payload, candidate_room_ids=CANDIDATES)
        with self.assertRaisesRegex(ValueError, "every candidate exactly once"):
            validate_map_grounded_decision_output(payload, candidate_room_ids=CANDIDATES)
        payload["predicted_room_id"] = "Room 999"
        with self.assertRaisesRegex(ValueError, "five candidates"):
            validate_map_grounded_top1_output(payload, candidate_room_ids=CANDIDATES)

    def test_compact_schema_requires_top1_and_prefix_recovery_locks_candidates(self) -> None:
        schema = build_map_grounded_decision_schema(
            candidate_room_ids=CANDIDATES,
            context_room_ids=CANDIDATES,
            view_ids=[f"view_{index}" for index in range(8)],
        )
        self.assertIn("predicted_room_id", schema["required"])
        self.assertEqual(
            recover_map_grounded_top1_prefix(
                '{"predicted_room_id":"Room 8","confidence":0.9',
                candidate_room_ids=CANDIDATES,
            ),
            "Room 8",
        )
        self.assertIsNone(
            recover_map_grounded_top1_prefix(
                '{"predicted_room_id":"Room 999"',
                candidate_room_ids=CANDIDATES,
            )
        )

    def test_compact_reasoning_validator_checks_all_candidates_and_rotation_winner(self) -> None:
        graph = mini_room_graph()
        views = [{"view_id": f"view_{index}", "heading": heading} for index, heading in enumerate(HEADINGS)]
        payload = compact_reasoning_output()
        validate_map_grounded_compact_reasoning_output(
            payload,
            candidate_room_ids=CANDIDATES,
            room_graph=graph,
            views=views,
        )
        payload["candidate_rotation_scores"][0]["east"] = 10.0
        with self.assertRaisesRegex(ValueError, "strongest candidate rotation score"):
            validate_map_grounded_compact_reasoning_output(
                payload,
                candidate_room_ids=CANDIDATES,
                room_graph=graph,
                views=views,
            )

    def test_run_once_recovers_malformed_compact_json_without_second_vlm_call(self) -> None:
        graph = mini_room_graph()
        requests = []
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=lambda request: requests.append(request)
                or {"output_text": '{"predicted_room_id":"Room 8","confidence":0.9'},
                validation_retries=2,
            )
            result = experiment.run_once(
                observation=observation,
                manifest={"heading_mode": "museum"},
                candidate_room_ids=CANDIDATES,
                semantic_map_path=semantic_map,
            )
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["result"]["predicted_room_id"], "Room 8")
        self.assertEqual(result["top1_source"], "json_prefix_recovery")
        self.assertFalse(result["spatial_reasoning_complete"])
        self.assertFalse(result["ranking_complete"])
        self.assertFalse(result["winning_hypothesis_audit_complete"])

    def test_run_once_does_not_issue_vlm_repair_when_compact_top1_is_missing(self) -> None:
        graph = mini_room_graph()
        requests = []
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=lambda request: requests.append(request) or {"output_text": "{}"},
                validation_retries=3,
            )
            with self.assertRaisesRegex(ValueError, "five candidates"):
                experiment.run_once(
                    observation=observation,
                    manifest={"heading_mode": "museum"},
                    candidate_room_ids=CANDIDATES,
                    semantic_map_path=semantic_map,
                )
        self.assertEqual(len(requests), 1)

    def test_run_once_parses_and_validates_one_response(self) -> None:
        graph = mini_room_graph()
        requests = []
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=lambda request: requests.append(request)
                or {
                    "output_text": json.dumps(compact_reasoning_output()),
                    "usage": {"input_tokens": 100},
                },
            )
            result = experiment.run_once(
                observation=observation,
                manifest={"heading_mode": "museum"},
                candidate_room_ids=CANDIDATES,
                semantic_map_path=semantic_map,
            )
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["logical_vlm_call_count"], 1)
        self.assertEqual(result["top1_source"], "compact_spatial_reasoning_json")
        self.assertTrue(result["spatial_reasoning_complete"])
        self.assertEqual(result["audit_warnings"], [])
        self.assertEqual(result["result"]["predicted_room_id"], "Room 8")

    def test_run_once_retries_once_after_local_validation_error(self) -> None:
        graph = mini_room_graph()
        responses = [
            {"output_text": json.dumps({"predicted_room_id": "Room 8"})},
            {"output_text": json.dumps(valid_output(graph))},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=lambda _: responses.pop(0),
                validation_retries=1,
                compact_decision_output=False,
            )
            result = experiment.run_once(
                observation=observation,
                manifest={"heading_mode": "museum"},
                candidate_room_ids=CANDIDATES,
                semantic_map_path=semantic_map,
            )
        self.assertEqual(result["validation_attempt_count"], 2)
        self.assertEqual(len(result["validation_errors_before_success"]), 1)
        self.assertEqual(result["result"]["predicted_room_id"], "Room 8")

    def test_run_once_falls_back_to_minimal_schema_after_http_400(self) -> None:
        graph = mini_room_graph()
        requests = []

        def response_client(request: dict) -> dict:
            requests.append(request)
            if len(requests) == 1:
                raise RuntimeError("Model API request failed with HTTP 400: invalid argument")
            return {"output_text": json.dumps(valid_output(graph))}

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=response_client,
                validation_retries=0,
                compact_decision_output=False,
            )
            result = experiment.run_once(
                observation=observation,
                manifest={"heading_mode": "museum"},
                candidate_room_ids=CANDIDATES,
                semantic_map_path=semantic_map,
            )
        self.assertEqual(len(requests), 2)
        self.assertNotEqual(requests[0]["text"]["format"]["schema"], {"type": "object"})
        self.assertEqual(requests[1]["text"]["format"]["schema"], {"type": "object"})
        self.assertEqual(result["structured_output_mode"], "gemini_minimal_object_schema_after_http_400")
        self.assertEqual(result["result"]["predicted_room_id"], "Room 8")

    def test_preflight_rejects_non_museum_heading_and_non_five_candidates(self) -> None:
        graph = mini_room_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            observation = build_observation(directory)
            semantic_map = directory / "semantic.png"
            Image.new("RGB", (200, 200), (250, 250, 250)).save(semantic_map)
            experiment = MapGroundedSpatialAlignmentExperiment(
                room_graph=graph,
                response_client=lambda _: {"output_text": "{}"},
            )
            with self.assertRaisesRegex(ValueError, "heading_mode=museum"):
                experiment.build_request_body(
                    observation=observation,
                    manifest={"heading_mode": "graph"},
                    candidate_room_ids=CANDIDATES,
                    semantic_map_path=semantic_map,
                )
            with self.assertRaisesRegex(ValueError, "exactly five"):
                experiment.build_request_body(
                    observation=observation,
                    manifest={"heading_mode": "museum"},
                    candidate_room_ids=CANDIDATES[:4],
                    semantic_map_path=semantic_map,
                )


if __name__ == "__main__":
    unittest.main()
