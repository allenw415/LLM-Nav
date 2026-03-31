from __future__ import annotations

from .models import RoomGroundingEntry, SourcePanoResolution


def build_grounding_template(room_graph: dict[str, dict]) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for room_id, node in room_graph.items():
        aliases = list(node.get("aliases") or [])
        anchor_entities = []
        title = node.get("title")
        category = node.get("category")
        if isinstance(title, str) and title:
            anchor_entities.append(title)
        if isinstance(category, str) and category:
            anchor_entities.append(category)

        entry = RoomGroundingEntry(
            room_id=room_id,
            floor=str(node.get("floor", "unknown")),
            aliases=aliases,
            anchor_entities=anchor_entities,
            notes="Fill pano_ids after manual alignment.",
        )
        entries[room_id] = entry.to_dict()
    return entries


class GroundingIndex:
    def __init__(self, grounding: dict[str, dict]):
        self._grounding = grounding

    def room_entry(self, room_id: str) -> dict | None:
        return self._grounding.get(room_id)

    def primary_pano_for_room(self, room_id: str) -> str | None:
        entry = self._grounding.get(room_id)
        if not entry:
            return None
        pano_ids = entry.get("pano_ids", [])
        if not pano_ids:
            return None
        return str(pano_ids[0])


class SourcePanoResolver:
    """
    Initialization-only resolver: source room -> representative pano.
    """

    def __init__(self, grounding_index: GroundingIndex):
        self.grounding_index = grounding_index

    def resolve(self, source_room_id: str) -> SourcePanoResolution:
        entry = self.grounding_index.room_entry(source_room_id) or {}
        candidate_pano_ids = [str(pano_id) for pano_id in entry.get("pano_ids", [])]
        pano_id = candidate_pano_ids[0] if candidate_pano_ids else None
        return SourcePanoResolution(
            source_room_id=source_room_id,
            pano_id=pano_id,
            candidate_pano_ids=candidate_pano_ids,
            resolution_method="representative_grounding",
        )
