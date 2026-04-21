from __future__ import annotations

import unittest

from st_nav_data.memory_localization import (
    aggregate_room_scores,
    cosine_similarity,
    deduplicate_candidates_by_pano,
    group_metadata_items_by_pano,
    is_valid_room_id,
    predict_room_from_candidates,
    resolve_siglip2_model_name,
    select_query_capture_records,
)


class MemoryLocalizationTests(unittest.TestCase):
    def test_resolve_siglip2_model_name_supports_aliases(self) -> None:
        self.assertEqual(resolve_siglip2_model_name("siglip2"), "google/siglip2-base-patch16-224")
        self.assertEqual(resolve_siglip2_model_name("siglip2-so400m"), "google/siglip2-so400m-patch14-384")

    def test_cosine_similarity_handles_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_group_metadata_items_by_pano_sorts_capture_order(self) -> None:
        groups = group_metadata_items_by_pano(
            [
                {"memory_index": 1, "pano_id": "p1", "room_id": "Room 7", "capture_index": 1, "capture_path": "b"},
                {"memory_index": 0, "pano_id": "p1", "room_id": "Room 7", "capture_index": 0, "capture_path": "a"},
                {"memory_index": 2, "pano_id": "p2", "room_id": "Room 8", "capture_index": 0, "capture_path": "c"},
                {"memory_index": 3, "pano_id": "p3", "room_id": "null", "capture_index": 0, "capture_path": "d"},
            ]
        )

        self.assertEqual([group["pano_id"] for group in groups], ["p1", "p2"])
        self.assertEqual([capture["memory_index"] for capture in groups[0]["captures"]], [0, 1])

    def test_is_valid_room_id_rejects_string_null(self) -> None:
        self.assertTrue(is_valid_room_id("Room 24"))
        self.assertFalse(is_valid_room_id("null"))
        self.assertFalse(is_valid_room_id(""))

    def test_deduplicate_candidates_by_pano_keeps_best_match(self) -> None:
        candidates = deduplicate_candidates_by_pano(
            [
                {"candidate_pano_id": "p1", "candidate_capture_index": 1, "room_id": "Room 24", "score": 0.8},
                {"candidate_pano_id": "p2", "candidate_capture_index": 0, "room_id": "Room 26", "score": 0.7},
                {"candidate_pano_id": "p1", "candidate_capture_index": 0, "room_id": "Room 24", "score": 0.9},
            ]
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["candidate_pano_id"], "p1")
        self.assertAlmostEqual(candidates[0]["score"], 0.9)

    def test_select_query_capture_records_supports_even_spacing(self) -> None:
        captures = [{"capture_index": index} for index in range(8)]

        selected = select_query_capture_records(captures, query_view_count=4, selection="evenly-spaced")

        self.assertEqual([record["capture_index"] for record in selected], [0, 2, 5, 7])

    def test_aggregate_room_scores_sums_per_room(self) -> None:
        scores = aggregate_room_scores(
            [
                {"room_id": "Room 7", "score": 0.8},
                {"room_id": "Room 8", "score": 0.6},
                {"room_id": "Room 7", "score": 0.4},
            ]
        )

        self.assertEqual(list(scores.keys())[0], "Room 7")
        self.assertAlmostEqual(scores["Room 7"], 1.2)
        self.assertAlmostEqual(scores["Room 8"], 0.6)

    def test_predict_room_from_candidates_returns_normalized_confidence(self) -> None:
        predicted_room_id, confidence, room_scores = predict_room_from_candidates(
            [
                {"room_id": "Room 26", "score": 0.9},
                {"room_id": "Room 26", "score": 0.3},
                {"room_id": "Room 27", "score": 0.4},
            ]
        )

        self.assertEqual(predicted_room_id, "Room 26")
        self.assertAlmostEqual(confidence, 0.75)
        self.assertEqual(list(room_scores.keys())[:2], ["Room 26", "Room 27"])


if __name__ == "__main__":
    unittest.main()
