from __future__ import annotations

import random

from ..common.types import RoomGroundingEntry, SourcePanoResolution
from ..common.room_profiles import visual_profile_anchor_entities


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
        anchor_entities.extend(
            value for value in visual_profile_anchor_entities(node) if value not in anchor_entities
        )

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
    def __init__(self, grounding: dict[str, dict] | None = None, pano_to_room: dict | None = None):
        self._grounding = grounding or {}
        self._pano_to_room: dict[str, str] = {}
        self._room_to_panos: dict[str, list[str]] = {}
        for room_id, entry in self._grounding.items():
            if not isinstance(room_id, str) or not room_id:
                continue
            pano_ids = entry.get("pano_ids", []) if isinstance(entry, dict) else []
            for pano_id in pano_ids:
                if not isinstance(pano_id, str) or not pano_id:
                    continue
                self._set_pano_room(pano_id, room_id)
        if isinstance(pano_to_room, dict):
            mappings = pano_to_room.get("mappings", pano_to_room)
            if isinstance(mappings, dict):
                for pano_id, room_id in mappings.items():
                    if not isinstance(pano_id, str) or not pano_id:
                        continue
                    if not isinstance(room_id, str) or not room_id or room_id == "null":
                        continue
                    self._set_pano_room(pano_id, room_id)

    def _set_pano_room(self, pano_id: str, room_id: str) -> None:
        previous_room_id = self._pano_to_room.get(pano_id)
        if previous_room_id and previous_room_id != room_id:
            previous_pano_ids = self._room_to_panos.get(previous_room_id, [])
            self._room_to_panos[previous_room_id] = [
                existing_pano_id for existing_pano_id in previous_pano_ids if existing_pano_id != pano_id
            ]

        self._pano_to_room[pano_id] = room_id
        pano_ids = self._room_to_panos.setdefault(room_id, [])
        if pano_id not in pano_ids:
            pano_ids.append(pano_id)

    def room_entry(self, room_id: str) -> dict | None:
        entry = self._grounding.get(room_id)
        pano_ids = self.pano_ids_for_room(room_id)
        if not isinstance(entry, dict):
            if not pano_ids:
                return None
            return {
                "room_id": room_id,
                "pano_ids": pano_ids,
            }
        merged = dict(entry)
        if pano_ids:
            merged["pano_ids"] = pano_ids
        return merged

    def pano_ids_for_room(self, room_id: str) -> list[str]:
        return list(self._room_to_panos.get(room_id, []))

    def primary_pano_for_room(self, room_id: str) -> str | None:
        pano_ids = self.pano_ids_for_room(room_id)
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

    def __init__(self, grounding_index: GroundingIndex, rng: random.Random | None = None):
        self.grounding_index = grounding_index
        self.rng = rng

    def resolve(self, source_room_id: str) -> SourcePanoResolution:
        candidate_pano_ids = self.grounding_index.pano_ids_for_room(source_room_id)
        chooser = self.rng.choice if self.rng is not None else random.choice
        pano_id = chooser(candidate_pano_ids) if candidate_pano_ids else None
        return SourcePanoResolution(
            source_room_id=source_room_id,
            pano_id=pano_id,
            candidate_pano_ids=candidate_pano_ids,
            resolution_method="random_room_grounding",
        )
