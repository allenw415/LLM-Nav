import json
from pathlib import Path
from typing import Any, Dict


def flatten_panos(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input format expected:
      {
        "<floor>": {
          "<panoId>": {
            "panoID": "...",
            "lat": ...,
            "lng": ...,
            "links": [...]
            ...
          },
          ...
        },
        ...
      }

    Output:
      {
        "<panoId>": {
          "panoID": "...",
          "lat": ...,
          "lng": ...,
          "links": [...],
          "floor": "<floor>"
        },
        ...
      }
    """
    out: Dict[str, Any] = {}

    for floor, pano_dict in data.items():
        if not isinstance(pano_dict, dict):
            continue

        for pano_key, rec in pano_dict.items():
            if not isinstance(rec, dict):
                continue

            pano_id = rec.get("panoID") or rec.get("pano") or rec.get("pano_id") or pano_key

            out[pano_key] = {
                "panoID": pano_id,
                "lat": rec.get("lat"),
                "lng": rec.get("lng"),
                "links": rec.get("links", []),
                "floor": str(floor),
            }

    return out


if __name__ == "__main__":
    in_path = Path("extracted_panos.json")
    out_path = Path("flatten_panos.json")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    flat = flatten_panos(data)

    out_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} (count={len(flat)}) ✅")
