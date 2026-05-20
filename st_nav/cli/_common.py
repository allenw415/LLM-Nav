from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from st_nav.common.room_profiles import preferred_room_graph_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return payload


def render_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_text_if_requested(text: str, output_path: str | Path | None) -> None:
    if not output_path:
        return
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


@dataclass
class NormalizedArtifacts:
    artifacts_dir: Path
    room_graph: dict | None = None
    pano_graph: dict | None = None
    grounding: dict | None = None
    pano_room_grounding: dict | None = None


def load_normalized_artifacts(
    artifacts_dir: str | Path,
    *,
    room_graph: bool = False,
    pano_graph: bool = False,
    grounding: bool = False,
    pano_room_grounding: bool = False,
) -> NormalizedArtifacts:
    resolved_dir = resolve_project_path(artifacts_dir)
    loaded_room_graph = load_json(preferred_room_graph_path(resolved_dir)) if room_graph or grounding else None
    loaded_pano_room_grounding = (
        load_json(resolved_dir / "pano_room_grounding.json")
        if pano_room_grounding or grounding
        else None
    )
    loaded_grounding = None
    if grounding:
        from st_nav_data.pano_room_grounding import build_room_grounding_from_pano_room_mapping

        loaded_grounding = build_room_grounding_from_pano_room_mapping(
            loaded_room_graph or {},
            loaded_pano_room_grounding or {},
        )
    return NormalizedArtifacts(
        artifacts_dir=resolved_dir,
        room_graph=loaded_room_graph if room_graph else None,
        pano_graph=load_json(resolved_dir / "pano_graph.json") if pano_graph else None,
        grounding=loaded_grounding,
        pano_room_grounding=loaded_pano_room_grounding if pano_room_grounding else None,
    )
