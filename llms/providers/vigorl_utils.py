"""Utility functions for ViGORL multi-turn visual grounding in VWA."""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from PIL import Image, ImageDraw


# Matches <tool_call>...</tool_call> across newlines (ViGORL zoom tool).
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Matches <answer>...</answer> for ViGORL's final coordinate output.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_tool_coordinate(tool_text: str) -> Optional[Tuple[int, int]]:
    """Extract (x, y) from JSON inside a <tool_call> block."""
    try:
        payload = json.loads(tool_text.strip())
        coord = payload["arguments"]["coordinate"]
        return (int(coord[0]), int(coord[1]))
    except Exception:
        return None


def extract_answer_coord(response: str) -> Optional[Tuple[int, int]]:
    """Extract (x, y) from <answer>...</answer> if the content looks like coordinates.

    Returns None if the answer does not contain parseable coordinates (e.g. it is
    a text answer for a stop/QA action).
    """
    m = ANSWER_RE.search(response)
    if not m:
        return None
    content = m.group(1).strip()
    coord_match = re.search(r"(\d+)[,\s]+(\d+)", content)
    if coord_match:
        return (int(coord_match.group(1)), int(coord_match.group(2)))
    return None


def get_point_crop(
    img: Image.Image,
    pt: Tuple[int, int],
    offset: int = 182,
    crop_size: int = 672,
    draw_dot: bool = False,
) -> Image.Image:
    """Return a square crop of *img* centred on *pt*, resized to *crop_size*."""
    x, y = pt
    w, h = img.size
    left   = max(0, x - offset)
    top    = max(0, y - offset)
    right  = min(w, x + offset)
    bottom = min(h, y + offset)
    crop = img.crop((left, top, right, bottom))
    if draw_dot:
        draw = ImageDraw.Draw(crop)
        r = 6
        draw.ellipse(
            (x - left - r, y - top - r, x - left + r, y - top + r),
            fill="red", outline="white", width=2,
        )
    return crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)


def coord_to_som_id(
    coord: Tuple[int, int],
    obs_nodes_info: dict,
) -> Optional[str]:
    """Map a pixel coordinate to the best-matching SOM element ID.

    *obs_nodes_info* is the dict stored at
    ``trajectory[-1]["info"]["observation_metadata"]["image"]["obs_nodes_info"]``,
    mapping element_id -> (center_x, center_y, width, height) in viewport pixels.

    Strategy:
      1. Find all bounding boxes that contain the point; return the one with the
         smallest area (most specific match).
      2. If none contain the point, return the element whose centre is nearest.
    """
    if not obs_nodes_info:
        return None

    x, y = coord

    containing: list[Tuple[float, str]] = []
    for elem_id, (cx, cy, w, h) in obs_nodes_info.items():
        if cx - w / 2 <= x <= cx + w / 2 and cy - h / 2 <= y <= cy + h / 2:
            containing.append((w * h, str(elem_id)))

    if containing:
        containing.sort(key=lambda t: t[0])
        return containing[0][1]

    # Nearest-centre fallback.
    return str(min(
        obs_nodes_info,
        key=lambda eid: (obs_nodes_info[eid][0] - x) ** 2 + (obs_nodes_info[eid][1] - y) ** 2,
    ))
