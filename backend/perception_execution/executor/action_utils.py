def normalize_heading(heading: float) -> float:
    return heading % 360.0

def clamp_pitch(pitch: float, min_pitch: float, max_pitch: float) -> float:
    return max(min_pitch, min(max_pitch, pitch))

def clamp_zoom(zoom: int, min_zoom: int, max_zoom: int) -> int:
    return max(min_zoom, min(max_zoom, zoom))