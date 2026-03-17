import json
from pathlib import Path
from typing import Any, Dict


def minimize_pano_records(obj: Any) -> Any:
    """
    支援兩種常見格式：
    A) {"panos": {panoId: {...}, ...}, ...}
    B) {"0": {panoId: {...}, ...}, "1": {...}}  (你貼的看起來像這種)
    會把每個 pano 的欄位縮減為：panoID, lat, lng, links
    """
    if not isinstance(obj, dict):
        return obj

    # case B: 外層是 step/分組 key，內層是 pano dict
    # 例如 {"0": {panoId: {...}}, "1": {...}}
    out: Dict[str, Any] = {}
    all_values_are_pano_dicts = True

    for k, v in obj.items():
        if isinstance(v, dict) and _looks_like_pano_dict(v):
            out[k] = _minimize_pano_dict(v)
        else:
            all_values_are_pano_dicts = False
            out[k] = v

    # 如果外層不是 case B，就原樣回傳（避免誤傷）
    return out if all_values_are_pano_dicts else obj


def _looks_like_pano_dict(d: Dict[str, Any]) -> bool:
    # 粗略判斷：key 看起來像 panoId，value 是 dict 且有 lat/lng 或 links
    if not d:
        return False
    sample_key = next(iter(d.keys()))
    sample_val = d[sample_key]
    if not isinstance(sample_val, dict):
        return False
    return ("lat" in sample_val and "lng" in sample_val) or ("links" in sample_val) or ("panoID" in sample_val)


def _minimize_pano_dict(pano_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pano_key, rec in pano_dict.items():
        if not isinstance(rec, dict):
            continue
        out[pano_key] = {
            "panoID": rec.get("panoID") or rec.get("pano") or rec.get("pano_id") or pano_key,
            "lat": rec.get("lat"),
            "lng": rec.get("lng"),
            "links": rec.get("links", []),
        }
    return out


if __name__ == "__main__":
    in_path = Path("panos.json")
    out_path = Path("extracted_panos.json")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    minimized = minimize_pano_records(data)

    out_path.write_text(json.dumps(minimized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} ✅")
