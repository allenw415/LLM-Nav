from __future__ import annotations

from ..common.types import RoomGroundingEntry, SourcePanoResolution


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
    def __init__(self, grounding: dict[str, dict], pano_to_room: dict | None = None):
        self._grounding = grounding
        self._pano_to_room: dict[str, str] = {}
        for room_id, entry in grounding.items():
            if not isinstance(room_id, str) or not room_id:
                continue
            pano_ids = entry.get("pano_ids", []) if isinstance(entry, dict) else []
            for pano_id in pano_ids:
                if not isinstance(pano_id, str) or not pano_id:
                    continue
                self._pano_to_room[pano_id] = room_id
        if isinstance(pano_to_room, dict):
            mappings = pano_to_room.get("mappings", pano_to_room)
            if isinstance(mappings, dict):
                for pano_id, room_id in mappings.items():
                    if not isinstance(pano_id, str) or not pano_id:
                        continue
                    if not isinstance(room_id, str) or not room_id or room_id == "null":
                        continue
                    self._pano_to_room[pano_id] = room_id

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

    def room_for_pano(self, pano_id: str) -> str | None:
        room_id = self._pano_to_room.get(pano_id)
        if not isinstance(room_id, str) or not room_id:
            return None
        if room_id == "null":
            return None
        return room_id


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
