"""A Duck-style, Python-tool ARC-AGI-3 solver.

Unlike :mod:`my_agent2`, this module does not ask the model to fill a fixed
scene-analysis schema.  The model gets one ``python`` tool.  Every invocation
receives a compact view of the current board and transition history, and may
call ``action(...)`` from Python to inspect, experiment, or play a short,
validated sequence.

This is intended for the Kaggle/TAAF runner.  It is deliberately standalone:
the normal ARC-AGI-3-Agents package does not depend on TAAF.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import arcengine
import httpx
import numpy as np
import taaf.game
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
)
from PIL import Image
from taaf.solver import Solver

COLOR_SYMBOLS = "0123456789ABCDEF"
PALETTE = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153),
    3: (102, 102, 102), 4: (51, 51, 51), 5: (0, 0, 0),
    6: (229, 58, 163), 7: (255, 123, 204), 8: (249, 60, 49),
    9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48),
    15: (163, 86, 214),
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run a short Python program against the current game state. "
            "Use action([...]) inside code to execute real, legal game actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, default)))
    except ValueError:
        return default


class RunLogger:
    """Structured run log, enabled by default outside competition reruns."""

    def __init__(self, game: Any) -> None:
        # Development runs need their model-error chronology to be useful.  Keep
        # competition reruns quiet by default, while allowing either mode to be
        # overridden explicitly with MY_AGENT3_LOG_DIR (an empty value disables it).
        root = os.getenv("MY_AGENT3_LOG_DIR")
        if root is None and not _env_bool("TAAF_RUN_AS_SUBMISSION"):
            kaggle_working = Path("/kaggle/working")
            root = str(kaggle_working / "my_agent3_logs") if kaggle_working.is_dir() else "my_agent3_logs"
        self.file: Any | None = None
        self.include_boards = _env_bool("MY_AGENT3_LOG_BOARDS")
        self.stdout = _env_bool("MY_AGENT3_LOG_STDOUT")
        if not root:
            return
        game_id = str(
            getattr(game, "game_id", None) or getattr(game, "env_name", None) or "unknown"
        )
        safe_game_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in game_id)
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / f"{safe_game_id}.{int(time.time() * 1000)}.jsonl").open(
            "a", encoding="utf-8"
        )
        self.record("run_started", game_id=game_id)

    def record(self, event: str, **data: Any) -> None:
        payload = {"timestamp": time.time(), "event": event, **data}
        if self.file is not None:
            self.file.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
            self.file.flush()
        if self.stdout:
            preview = json.dumps(payload, default=str)[:2_000]
            # Python tool code is run under redirect_stdout().  Mirrored
            # diagnostics must not become accidental, often huge, model tool
            # output; keep them on the original process stream instead.
            print(f"[MYAGENT3 {event}] {preview}", file=sys.__stdout__, flush=True)

    def board(self, event: str, frame: "FrameView") -> None:
        payload: dict[str, Any] = {
            "shape": frame.shape,
            "level": frame.level,
            "step": frame.step,
            "component_count": len(frame.segmentation["nodes"]),
        }
        if self.include_boards:
            payload["ascii"] = frame.ascii
            payload["segmentation"] = frame.segmentation
        self.record(event, **payload)

    def close(self) -> None:
        if self.file is not None:
            self.record("run_finished")
            self.file.close()
            self.file = None


def _game_action(value: Any) -> arcengine.GameAction:
    """Resolve an enum, numeric id, or action name without accepting aliases."""
    if isinstance(value, arcengine.GameAction):
        return value
    if isinstance(value, int):
        return arcengine.GameAction.from_id(value)
    if isinstance(value, str):
        name = value.upper().strip()
        if name == "MOUSE":
            name = "ACTION6"
        try:
            return arcengine.GameAction.from_name(name)
        except (AttributeError, KeyError, ValueError):
            return arcengine.GameAction.from_id(int(name))
    raise TypeError(f"Unsupported action value: {value!r}")


def _action_name(value: Any) -> str:
    try:
        return _game_action(value).name
    except (TypeError, ValueError):
        return str(value)


def _board_from_state(state: Any) -> np.ndarray:
    board = np.asarray(state.frame.data, dtype=np.int16)
    # TAAF currently supplies a 2-D board.  This makes an accidental leading
    # animation/frame dimension harmless rather than silently flattening it.
    if board.ndim == 3:
        board = board[-1]
    if board.ndim != 2:
        raise ValueError(f"Expected a 2-D game board, received shape {board.shape}")
    return board


def _ascii(board: np.ndarray) -> str:
    return "\n".join(
        "".join(COLOR_SYMBOLS[int(cell)] if 0 <= int(cell) < 16 else "?" for cell in row)
        for row in board
    )


def _image_url(board: np.ndarray) -> str:
    height, width = board.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for color, value in PALETTE.items():
        rgb[board == color] = value
    # A 64px board loses object detail in many VLM preprocessors.
    image = Image.fromarray(rgb).resize((width * 8, height * 8), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _components(board: np.ndarray) -> dict[str, Any]:
    """Return compact components in O(board area), with bounded containment."""
    height, width = board.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    nodes: list[dict[str, Any]] = []
    for row in range(height):
        for col in range(width):
            if labels[row, col] >= 0:
                continue
            color = int(board[row, col])
            queue = deque([(row, col)])
            component_id = len(nodes)
            labels[row, col] = component_id
            pixels: set[tuple[int, int]] = set()
            ordered_pixels: list[tuple[int, int]] = []
            min_y = max_y = row
            min_x = max_x = col
            while queue:
                y, x = queue.popleft()
                pixels.add((y, x))
                ordered_pixels.append((y, x))
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_x, max_x = min(min_x, x), max(max_x, x)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < height and 0 <= nx < width
                        and labels[ny, nx] < 0 and int(board[ny, nx]) == color
                    ):
                        labels[ny, nx] = component_id
                        queue.append((ny, nx))
            # BFS starts at the component's top-left-most cell and has a fixed
            # neighbour order, so ordered_pixels is a deterministic shape key
            # without an O(component_size log component_size) sort.
            normal = [(y - min_y, x - min_x) for y, x in ordered_pixels]
            digest = hashlib.sha1(repr((color, normal)).encode()).hexdigest()[:12]
            boundary = [
                (y, x) for y, x in pixels
                if any((y + dy, x + dx) not in pixels for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            ]
            nodes.append({
                "id": len(nodes), "color": COLOR_SYMBOLS[color] if 0 <= color < 16 else "?",
                "color_id": color, "hash": digest, "pixels": len(pixels),
                "bbox": [min_y, min_x, max_y, max_x], "boundary": boundary[:128], "children": [],
            })

    # Every adjacent component pair appears on a horizontal or vertical grid
    # edge.  Scanning those edges avoids comparing every pair of objects.
    adjacency: set[tuple[int, int]] = set()
    for left, right in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        changed = left != right
        for first, second in zip(left[changed].tolist(), right[changed].tolist()):
            adjacency.add((min(first, second), max(first, second)))

    # Bounding-box containment is useful for small object scenes, but an O(n²)
    # relation is not acceptable for highly fragmented boards.  It is omitted
    # above this cap rather than delaying the first inference request.
    containment_cap = _env_int("MY_AGENT3_MAX_CONTAINMENT_NODES", 128)
    containment_truncated = len(nodes) > containment_cap
    if not containment_truncated:
        for child in nodes:
            y1, x1, y2, x2 = child["bbox"]
            containers = []
            for parent in nodes:
                if parent["id"] == child["id"]:
                    continue
                py1, px1, py2, px2 = parent["bbox"]
                if py1 <= y1 and px1 <= x1 and py2 >= y2 and px2 >= x2:
                    containers.append(parent)
            if containers:
                parent = min(containers, key=lambda item: item["pixels"])
                parent["children"].append(child["id"])
    return {
        "nodes": nodes,
        "adjacency_list": [list(edge) for edge in sorted(adjacency)],
        "containment_truncated": containment_truncated,
    }


def _component_brief(node: dict[str, Any]) -> dict[str, Any]:
    return {
        key: node[key]
        for key in ("id", "color", "pixels", "bbox", "hash")
    }


def _bbox_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Manhattan distance between component bounding-box centres, in grid cells."""
    ly1, lx1, ly2, lx2 = left["bbox"]
    ry1, rx1, ry2, rx2 = right["bbox"]
    return abs((ly1 + ly2) - (ry1 + ry2)) + abs((lx1 + lx2) - (rx1 + rx2))


def _transition_summary(
    before_board: np.ndarray,
    after_board: np.ndarray,
    before_segmentation: dict[str, Any],
    after_segmentation: dict[str, Any],
) -> dict[str, Any]:
    """Describe a transition compactly enough to be useful in a tool response."""
    changed = before_board != after_board
    coordinates = np.argwhere(changed)
    changed_count = int(len(coordinates))
    cell_summary: dict[str, Any] = {"count": changed_count}
    if changed_count:
        min_y, min_x = coordinates.min(axis=0).tolist()
        max_y, max_x = coordinates.max(axis=0).tolist()
        cell_summary["bbox"] = [int(min_y), int(min_x), int(max_y), int(max_x)]
        color_changes = Counter(
            (int(before_board[y, x]), int(after_board[y, x]))
            for y, x in coordinates
        )
        cell_summary["color_changes"] = [
            {
                "from": COLOR_SYMBOLS[before] if 0 <= before < len(COLOR_SYMBOLS) else "?",
                "to": COLOR_SYMBOLS[after] if 0 <= after < len(COLOR_SYMBOLS) else "?",
                "cells": count,
            }
            for (before, after), count in color_changes.most_common(12)
        ]
        sample_cap = _env_int("MY_AGENT3_CHANGE_CELL_SAMPLES", 24)
        cell_summary["samples"] = [
            {
                "row": int(y), "col": int(x),
                "from": COLOR_SYMBOLS[int(before_board[y, x])],
                "to": COLOR_SYMBOLS[int(after_board[y, x])],
            }
            for y, x in coordinates[:sample_cap]
        ]

    before_nodes = before_segmentation["nodes"]
    after_nodes = after_segmentation["nodes"]
    tracking_cap = _env_int("MY_AGENT3_MAX_TRACKED_COMPONENTS", 256)
    tracking_truncated = len(before_nodes) > tracking_cap or len(after_nodes) > tracking_cap
    moved: list[dict[str, Any]] = []
    appeared: list[dict[str, Any]] = []
    disappeared: list[dict[str, Any]] = []
    if not tracking_truncated:
        # A colour/shape/pixel-count match is a conservative identity cue.  It
        # handles translated players without pretending that local node ids are
        # persistent.  Ambiguous duplicates are paired by nearest position.
        before_groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
        after_groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
        for node in before_nodes:
            before_groups[(node["color_id"], node["hash"], node["pixels"])].append(node)
        for node in after_nodes:
            after_groups[(node["color_id"], node["hash"], node["pixels"])].append(node)

        unmatched_before = {node["id"] for node in before_nodes}
        unmatched_after = {node["id"] for node in after_nodes}
        before_by_id = {node["id"]: node for node in before_nodes}
        after_by_id = {node["id"]: node for node in after_nodes}
        for signature, prior_nodes in before_groups.items():
            remaining = list(after_groups.get(signature, []))
            for prior in prior_nodes:
                if not remaining:
                    break
                current = min(remaining, key=lambda node: _bbox_distance(prior, node))
                remaining.remove(current)
                unmatched_before.discard(prior["id"])
                unmatched_after.discard(current["id"])
                before_bbox, after_bbox = prior["bbox"], current["bbox"]
                row_delta = int(after_bbox[0] - before_bbox[0])
                col_delta = int(after_bbox[1] - before_bbox[1])
                if row_delta or col_delta:
                    moved.append({
                        "before_id": prior["id"], "after_id": current["id"],
                        "color": prior["color"], "pixels": prior["pixels"], "hash": prior["hash"],
                        "before_bbox": before_bbox, "after_bbox": after_bbox,
                        "row_delta": row_delta, "col_delta": col_delta,
                        "row_delta_note": "negative is up; positive is down",
                        "col_delta_note": "negative is left; positive is right",
                    })
        component_cap = _env_int("MY_AGENT3_COMPONENT_CHANGE_SAMPLES", 16)
        appeared = [_component_brief(after_by_id[node_id]) for node_id in sorted(unmatched_after)[:component_cap]]
        disappeared = [_component_brief(before_by_id[node_id]) for node_id in sorted(unmatched_before)[:component_cap]]

    return {
        "cells": cell_summary,
        "moved_components": moved[:_env_int("MY_AGENT3_MOVEMENT_SAMPLES", 16)],
        "appeared_components": appeared,
        "disappeared_components": disappeared,
        "tracking_truncated": tracking_truncated,
    }


def _region_array(
    board: np.ndarray, row: int, col: int, height: int, width: int,
) -> np.ndarray:
    """Validate and extract a deliberately small board crop."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (row, col, height, width)):
        raise TypeError("row, col, height, and width must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height * width > _env_int("MY_AGENT3_MAX_REGION_CELLS", 400):
        raise ValueError("requested region exceeds MY_AGENT3_MAX_REGION_CELLS")
    end_row, end_col = row + height, col + width
    if row < 0 or col < 0 or end_row > board.shape[0] or end_col > board.shape[1]:
        raise ValueError(f"region [{row}:{end_row}, {col}:{end_col}] is outside board {board.shape}")
    return board[row:end_row, col:end_col]


def _region_view(board: np.ndarray, row: int, col: int, height: int, width: int) -> dict[str, Any]:
    region = _region_array(board, row, col, height, width)
    return {
        "bbox": [row, col, row + height - 1, col + width - 1],
        "shape": [height, width],
        "row_labels": list(range(row, row + height)),
        "column_labels": list(range(col, col + width)),
        "ascii": _ascii(region),
    }


def _clamped_region_view(
    board: np.ndarray, row: int, col: int, height: int, width: int,
) -> dict[str, Any]:
    """Return the requested view after clipping it to board and output limits."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (row, col, height, width)):
        raise TypeError("row, col, height, and width must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    requested = {"row": row, "col": col, "height": height, "width": width}
    max_cells = _env_int("MY_AGENT3_MAX_REGION_CELLS", 400)
    height, width = min(height, board.shape[0]), min(width, board.shape[1])
    if height * width > max_cells:
        aspect = width / height
        height = max(1, min(height, int((max_cells / aspect) ** 0.5)))
        width = max(1, min(width, max_cells // height))
    top = min(max(row, 0), board.shape[0] - height)
    left = min(max(col, 0), board.shape[1] - width)
    view = _region_view(board, top, left, height, width)
    view["requested"] = requested
    view["clamped"] = requested != {
        "row": top, "col": left, "height": height, "width": width,
    }
    return view


@dataclass
class EdgeSignalTracker:
    """Learn when a long, thin edge component behaves like a HUD indicator.

    An edge location is ignored for state fingerprints only after its component
    count changes monotonically for several actions. Touching an edge alone
    never makes a region a HUD.
    """

    candidates: dict[tuple[int, int, int, int, str], dict[str, Any]] = field(default_factory=dict)
    min_progress_steps: int = field(
        default_factory=lambda: _env_int("MY_AGENT3_HUD_CONFIRM_STEPS", 3)
    )

    @staticmethod
    def _discover(frame: FrameView) -> dict[tuple[int, int, int, int, str], dict[str, Any]]:
        height, width = frame.shape
        found = {}
        for node in frame.segmentation["nodes"]:
            r1, c1, r2, c2 = map(int, node["bbox"])
            box_height, box_width = r2 - r1 + 1, c2 - c1 + 1
            horizontal = box_height <= 4 and box_width >= max(8, width // 4)
            vertical = box_width <= 4 and box_height >= max(8, height // 4)
            touches_edge = r1 == 0 or c1 == 0 or r2 == height - 1 or c2 == width - 1
            if touches_edge and (horizontal or vertical):
                key = (r1, c1, r2, c2, str(node["color"]))
                found[key] = {"bbox": [r1, c1, r2, c2], "color": str(node["color"])}
        return found

    @staticmethod
    def _count(frame: FrameView, bbox: list[int], color: str) -> int:
        r1, c1, r2, c2 = bbox
        rows = frame.ascii.splitlines()
        return sum(row[c1:c2 + 1].count(color) for row in rows[r1:r2 + 1])

    def observe(
        self, before: FrameView, after: FrameView, summary: dict[str, Any],
        tracked_entity_visible: bool = False,
    ) -> dict[str, Any]:
        for key, candidate in self._discover(before).items():
            self.candidates.setdefault(key, {
                **candidate, "progress_steps": 0, "last_direction": 0,
                "verified": False, "observations": 0, "nonmoving_indicator_observations": 0,
            })
        changed_bbox = (summary.get("cells") or {}).get("bbox")
        changed = bool((summary.get("cells") or {}).get("count", 0))
        for candidate in self.candidates.values():
            old_count = self._count(before, candidate["bbox"], candidate["color"])
            new_count = self._count(after, candidate["bbox"], candidate["color"])
            delta = new_count - old_count
            if not delta:
                continue
            candidate["observations"] += 1
            direction = 1 if delta > 0 else -1
            candidate["progress_steps"] = (
                candidate["progress_steps"] + 1
                if candidate["last_direction"] == direction else 1
            )
            candidate["last_direction"] = direction
        hud_only = False
        if changed and changed_bbox:
            cr1, cc1, cr2, cc2 = changed_bbox
            for candidate in self.candidates.values():
                r1, c1, r2, c2 = candidate["bbox"]
                inside = r1 <= cr1 <= cr2 <= r2 and c1 <= cc1 <= cc2 <= c2
                if inside and not summary.get("moved_components") and tracked_entity_visible:
                    candidate["nonmoving_indicator_observations"] += 1
                    if (
                        candidate["progress_steps"] >= self.min_progress_steps
                        and candidate["nonmoving_indicator_observations"] >= 2
                    ):
                        candidate["verified"] = True
                if candidate["verified"] and inside and not summary.get("moved_components"):
                    hud_only = True
                    break
        return {
            "classification": "likely_hud_only" if hud_only else "gameplay_or_uncertain",
            "gameplay_changed": bool(changed and not hud_only),
            "candidate_indicators": [dict(value) for value in self.candidates.values()],
        }

    def ignored_regions(self) -> list[list[int]]:
        return [value["bbox"] for value in self.candidates.values() if value["verified"]]




def _relative_pattern_mismatches(first: np.ndarray, second: np.ndarray) -> int:
    """Compare equality structure while allowing a consistent colour remap."""
    first_to_second: dict[int, int] = {}
    second_to_first: dict[int, int] = {}
    mismatches = 0
    for left, right in zip(first.flat, second.flat):
        left_id, right_id = int(left), int(right)
        if (
            left_id in first_to_second and first_to_second[left_id] != right_id
        ) or (
            right_id in second_to_first and second_to_first[right_id] != left_id
        ):
            mismatches += 1
            continue
        first_to_second[left_id] = right_id
        second_to_first[right_id] = left_id
    return mismatches


def _compare_regions(
    board: np.ndarray,
    row1: int,
    col1: int,
    row2: int,
    col2: int,
    height: int,
    width: int,
    rotations: bool = True,
    reflections: bool = True,
) -> dict[str, Any]:
    """Compare two crops under exact and colour-remapped pattern matching."""
    first = _region_array(board, row1, col1, height, width)
    second = _region_array(board, row2, col2, height, width)
    variants: list[tuple[str, np.ndarray]] = [("identity", second)]
    if rotations:
        variants.extend((f"rotate_{degrees}", np.rot90(second, turns)) for turns, degrees in ((1, 90), (2, 180), (3, 270)))
    if reflections:
        variants.append(("flip_left_right", np.fliplr(second)))
        variants.append(("flip_up_down", np.flipud(second)))
    if rotations and reflections:
        variants.extend(
            (f"rotate_{degrees}_flip_left_right", np.fliplr(np.rot90(second, turns)))
            for turns, degrees in ((1, 90), (2, 180), (3, 270))
        )

    exact_options: list[tuple[int, str]] = []
    relative_options: list[tuple[int, str]] = []
    for name, candidate in variants:
        if candidate.shape != first.shape:
            continue
        exact_options.append((int(np.count_nonzero(first != candidate)), name))
        relative_options.append((_relative_pattern_mismatches(first, candidate), name))
    if not exact_options:
        raise ValueError("no requested transform preserves the region dimensions")
    exact_mismatches, exact_transform = min(exact_options)
    relative_mismatches, relative_transform = min(relative_options)
    cells = int(first.size)
    return {
        "first": _region_view(board, row1, col1, height, width),
        "second": _region_view(board, row2, col2, height, width),
        "exact": {
            "best_transform": exact_transform,
            "matching_cells": cells - exact_mismatches,
            "mismatched_cells": exact_mismatches,
            "equal": exact_mismatches == 0,
        },
        "relative_pattern": {
            "best_transform": relative_transform,
            "matching_cells": cells - relative_mismatches,
            "mismatched_cells": relative_mismatches,
            "equal": relative_mismatches == 0,
            "meaning": "same equality pattern with a consistent colour remap",
        },
    }


def _compact_transition_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep a fresh turn informative without replaying verbose tool output."""
    summary = result.get("change_summary") or {}
    cells = summary.get("cells") or {}
    prediction_check = result.get("prediction_check")
    return {
        key: result.get(key)
        for key in (
            "action", "board_changed", "gameplay_changed", "hud_assessment",
            "level_completed", "game_over", "done", "valid_actions", "actions_taken",
        )
    } | {
        "change": {
            "cells": {key: cells[key] for key in ("count", "bbox", "color_changes") if key in cells},
            "moved_components": [
                {
                    key: component[key]
                    for key in (
                        "color", "pixels", "hash", "before_bbox", "after_bbox", "row_delta", "col_delta",
                    )
                    if key in component
                }
                for component in summary.get("moved_components", [])
            ],
            "appeared_count": len(summary.get("appeared_components", [])),
            "disappeared_count": len(summary.get("disappeared_components", [])),
            "tracking_truncated": summary.get("tracking_truncated", False),
        },
    } | (
        {"prediction_check": {
            key: prediction_check.get(key)
            for key in ("action", "expected", "matched", "observed")
        }}
        if prediction_check is not None else {}
    )


def _repair_trailing_tool_artifact(code: str) -> str | None:
    """Repair only the recurring unmatched ``},`` suffix emitted by some servers."""
    stripped = code.rstrip()
    for suffix in ("},", "}"):
        if not stripped.endswith(suffix):
            continue
        repaired = stripped[: -len(suffix)].rstrip()
        if repaired.endswith(")"):
            return repaired
    return None

def _uses_direct_ascii(code: str) -> bool:
    """Reject unbounded board-string access from model-authored code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        (isinstance(node, ast.Attribute) and node.attr == "ascii")
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "ascii"
        )
        for node in ast.walk(tree)

    )



def _motion_member_key(component: dict[str, Any]) -> tuple[str, int, str]:
    """Stable-enough identifier for a shape-preserving translated component."""
    return (
        str(component["color"]),
        int(component["pixels"]),
        str(component.get("hash", "")),
    )


def _motion_member_brief(member: tuple[str, int, str]) -> dict[str, Any]:
    color, pixels, shape_hash = member
    return {"color": color, "pixels": pixels, "hash": shape_hash}


@dataclass
class MotionHypothesisTracker:
    """Accumulate conservative, action-dependent translation hypotheses.

    Component ids belong to one segmentation only, so they cannot be used as
    tracks.  Instead we use the existing exact colour/shape/pixel match from a
    transition summary.  This intentionally excludes components that merely
    grow, shrink, appear, or disappear (for example, a progress indicator).
    """

    max_candidates: int = field(
        default_factory=lambda: _env_int("MY_AGENT3_MAX_MOTION_HYPOTHESES", 64)
    )
    action_trials: Counter[str] = field(default_factory=Counter)
    candidate_trials: dict[tuple[tuple[str, int, str], ...], Counter[str]] = field(
        default_factory=dict
    )
    candidates: dict[tuple[tuple[str, int, str], ...], Counter[tuple[str, int, int]]] = field(
        default_factory=dict
    )
    non_translation_effects: dict[tuple[tuple[str, int, str], ...], Counter[str]] = field(
        default_factory=dict
    )
    non_translation_examples: dict[
        tuple[tuple[str, int, str], ...], dict[str, list[dict[str, Any]]]
    ] = field(default_factory=dict)

    def observe(
        self,
        action: str,
        transition: dict[str, Any],
        before_segmentation: dict[str, Any],
    ) -> None:
        """Record every exact translation and the co-moving groups it implies."""
        self.action_trials[action] += 1
        if transition.get("tracking_truncated"):
            return

        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for component in transition.get("moved_components", []):
            row_delta = component.get("row_delta")
            col_delta = component.get("col_delta")
            if not isinstance(row_delta, int) or not isinstance(col_delta, int):
                continue
            if not (row_delta or col_delta):
                continue
            groups[(row_delta, col_delta)].append(component)

        moved_members = {
            _motion_member_key(component)
            for components in groups.values()
            for component in components
        }
        visible_nodes: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for node in before_segmentation["nodes"]:
            visible_nodes[_motion_member_key(node)].append(node)

        # A learned candidate that is visible before an action but does not
        # translate afterwards is valuable route evidence.  It may mean a wall,
        # a boundary, or a conditional control, so do not overstate it as a
        # definite collision.  This is deliberately evaluated before adding
        # candidates from the current action.
        for members in self.candidates:
            if not all(visible_nodes.get(member) for member in members):
                continue
            self.candidate_trials.setdefault(members, Counter())[action] += 1
            if any(member in moved_members for member in members):
                continue
            effects = self.non_translation_effects.setdefault(members, Counter())
            effects[action] += 1
            examples = self.non_translation_examples.setdefault(members, {})
            examples[action] = [
                {
                    "color": node["color"],
                    "pixels": node["pixels"],
                    "bbox": node["bbox"],
                }
                for member in members
                for node in visible_nodes[member][:1]
            ]

        # Store both each part and a group when several parts move together.
        # A group can represent a multicolour player without assuming that one
        # particular colour is the entity's "real" identity.
        for (row_delta, col_delta), components in groups.items():
            member_groups = [
                (member,)
                for member in (_motion_member_key(component) for component in components)
            ]
            if len(components) > 1:
                member_groups.append(tuple(sorted(_motion_member_key(component) for component in components)))
            for members in member_groups:
                if members not in self.candidates and len(self.candidates) >= self.max_candidates:
                    continue
                is_new_candidate = members not in self.candidates
                effects = self.candidates.setdefault(members, Counter())
                if is_new_candidate:
                    self.candidate_trials.setdefault(members, Counter())[action] += 1
                effects[(action, row_delta, col_delta)] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return compact evidence, ranked without claiming certainty too early."""
        candidates: list[dict[str, Any]] = []
        for members, effects in self.candidates.items():
            action_effects = []
            total_support = 0
            for action in sorted({action for action, _, _ in effects}):
                options = [
                    (count, row_delta, col_delta)
                    for (effect_action, row_delta, col_delta), count in effects.items()
                    if effect_action == action
                ]
                support, row_delta, col_delta = max(options)
                total_support += support
                trials = self.candidate_trials.get(members, Counter())[action]
                consistency = support / trials if trials else 0.0
                # One observation is useful but not proof; repeated, consistent
                # probes asymptotically approach 1.0.
                confidence = consistency * support / (support + 1)
                action_effects.append({
                    "action": action,
                    "row_delta": row_delta,
                    "col_delta": col_delta,
                    "observations": support,
                    "action_trials": trials,
                    "confidence": round(confidence, 2),
                })
            candidate_confidence = max(
                (effect["confidence"] for effect in action_effects), default=0.0
            )
            member_label = "+".join(
                f"{color}{pixels}:{shape_hash[:6]}"
                for color, pixels, shape_hash in members
            )
            candidates.append({
                "candidate_id": (
                    f"group:{member_label}" if len(members) > 1 else f"component:{member_label}"
                ),
                "kind": "co_moving_group" if len(members) > 1 else "component",
                "members": [_motion_member_brief(member) for member in members],
                "translation_observations": total_support,
                "confidence": candidate_confidence,
                "action_effects": action_effects,
                "non_translation_effects": [
                    {
                        "action": action,
                        "observations": observations,
                        "latest_member_bboxes": self.non_translation_examples
                        .get(members, {})
                        .get(action, []),
                        "note": (
                            "Candidate was visible but did not translate; this can indicate "
                            "a wall, boundary, or conditional/non-movement action."
                        ),
                    }
                    for action, observations in sorted(
                        self.non_translation_effects.get(members, {}).items()
                    )
                ],
            })

        candidates.sort(
            key=lambda candidate: (
                candidate["confidence"],
                candidate["translation_observations"],
                len(candidate["members"]),
            ),
            reverse=True,
        )
        # A composite candidate conveys the same evidence as each of its
        # individual members. Prefer it in the compact turn state so the model
        # sees the player-like whole before its redundant parts.
        selected: list[dict[str, Any]] = []
        covered_members: set[tuple[str, int, str]] = set()
        for candidate in candidates:
            members = {
                (member["color"], member["pixels"], member["hash"])
                for member in candidate["members"]
            }
            if candidate["kind"] == "component" and members <= covered_members:
                continue
            selected.append(candidate)
            if candidate["kind"] == "co_moving_group":
                covered_members.update(members)
            if len(selected) >= _env_int("MY_AGENT3_MOTION_HYPOTHESIS_SAMPLES", 8):
                break
        return {
            "action_trials": dict(sorted(self.action_trials.items())),
            "candidates": selected,
            "note": (
                "Only exact shape-preserving translations are candidates; "
                "in-place transformations are deliberately excluded."
            ),
        }


@dataclass
class FrameView:
    ascii: str
    segmentation: dict[str, Any]
    shape: tuple[int, int]
    step: int | None
    level: int | None

    def __repr__(self) -> str:
        # The default dataclass repr includes the entire ASCII grid and
        # segmentation. Models occasionally print the object while orienting;
        # keep that harmless without hiding the raw grid from explicit access.
        return (
            f"FrameView(shape={self.shape}, level={self.level}, step={self.step}, "
            f"component_count={len(self.segmentation['nodes'])}; "
            "use .ascii, view_region(), or .segmentation for details)"
        )


@dataclass
class TransitionView:
    action: str
    before_frame: FrameView
    after_frame: FrameView
    result: dict[str, Any]

    @property
    def frame(self) -> FrameView:
        return self.after_frame


@dataclass
class ExplorationLedger:
    """Record generic action coverage without selecting actions for the model."""

    state_visits: Counter[str] = field(default_factory=Counter)
    state_trials: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    level_trials: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    state_outcomes: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    level_outcomes: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def _state_id(frame: FrameView, ignored_regions: list[list[int]] | None = None) -> str:
        rows = [list(row) for row in frame.ascii.splitlines()]
        for r1, c1, r2, c2 in ignored_regions or []:
            for row in range(max(0, r1), min(len(rows), r2 + 1)):
                for col in range(max(0, c1), min(len(rows[row]), c2 + 1)):
                    rows[row][col] = "*"
        fingerprint = "\n".join("".join(row) for row in rows)
        digest = hashlib.blake2b(fingerprint.encode("utf-8"), digest_size=6).hexdigest()
        return f"L{frame.level if frame.level is not None else '?'}:{digest}"

    @staticmethod
    def _level_id(frame: FrameView) -> str:
        return str(frame.level) if frame.level is not None else "unknown"

    @staticmethod
    def _brief_effect(result: dict[str, Any]) -> dict[str, Any]:
        summary = result.get("change_summary") or {}
        cells = summary.get("cells") or {}
        return {
            "board_changed": bool(result.get("board_changed")),
            "gameplay_changed": bool(result.get("gameplay_changed")),
            "changed_cells": int(cells.get("count", 0)),
            "moved": [
                {
                    key: component[key]
                    for key in ("color", "pixels", "row_delta", "col_delta")
                    if key in component
                }
                for component in summary.get("moved_components", [])[:2]
            ],
            "level_completed": bool(result.get("level_completed")),
            "game_over": bool(result.get("game_over")),
        }

    def visit(self, frame: FrameView, ignored_regions: list[list[int]] | None = None) -> None:
        self.state_visits[self._state_id(frame, ignored_regions)] += 1

    def observe(
        self,
        before: FrameView,
        action: str,
        result: dict[str, Any],
        after: FrameView,
        ignored_regions: list[list[int]] | None = None,
    ) -> None:
        state_id, level_id = self._state_id(before, ignored_regions), self._level_id(before)
        effect = self._brief_effect(result)
        self.state_trials[state_id][action] += 1
        self.level_trials[level_id][action] += 1
        self.state_outcomes[(state_id, action)] = effect
        self.level_outcomes[(level_id, action)] = effect
        self.visit(after, ignored_regions)

    @staticmethod
    def _tested_actions(
        trials: Counter[str],
        outcomes: dict[tuple[str, str], dict[str, Any]],
        scope: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "action": action,
                "trials": int(trials[action]),
                "latest_effect": outcomes.get((scope, action), {}),
            }
            for action in sorted(trials)
        ]

    def snapshot(
        self, frame: FrameView, valid_actions: list[str],
        ignored_regions: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        """Summarize state-action coverage, ignoring verified HUD regions."""
        state_id, level_id = self._state_id(frame, ignored_regions), self._level_id(frame)
        state_trials = self.state_trials.get(state_id, Counter())
        level_trials = self.level_trials.get(level_id, Counter())
        explorable = sorted(action for action in valid_actions if action != "RESET")
        return {
            "current_state": {
                "id": state_id,
                "visits": int(self.state_visits.get(state_id, 0)),
                "tested_actions": self._tested_actions(
                    state_trials, self.state_outcomes, state_id
                ),
            },
            "this_level": {
                "tested_actions": self._tested_actions(
                    level_trials, self.level_outcomes, level_id
                ),
                "untested_legal_actions": [
                    action for action in explorable if not level_trials.get(action, 0)
                ],
            },
            "note": (
                "Coverage is observational, not a command: untested actions exclude RESET "
                "and should be chosen only when they are a useful discriminating probe."
            ),
        }
    def has_tested_this_level(self, frame: FrameView, action: str) -> bool:
        return bool(self.level_trials.get(self._level_id(frame), Counter()).get(action, 0))



@dataclass
class PredictionLedger:
    """Check model-stated action expectations and retain their evidence."""

    max_recent: int = field(
        default_factory=lambda: _env_int("MY_AGENT3_PREDICTION_SAMPLES", 8)
    )
    pending: tuple[str, dict[str, Any]] | None = None
    replan_required: dict[str, Any] | None = None
    recent: deque[dict[str, Any]] = field(default_factory=deque)
    rule_stats: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def clear_pending(self) -> None:
        self.pending = None

    def set_pending(self, action: str, expected: dict[str, Any]) -> dict[str, Any]:
        self.pending = (action, expected)
        return {"action": action, "expected": expected, "status": "registered_for_next_action"}

    def take_pending(self, action: str) -> dict[str, Any] | None:
        pending = self.pending
        self.pending = None
        if pending is None:
            return None
        expected_action, expected = pending
        if expected_action != action:
            raise ValueError(
                f"prediction was registered for {expected_action}, not requested {action}"
            )
        return expected

    @staticmethod
    def _observed_effect(result: dict[str, Any]) -> dict[str, Any]:
        summary = result.get("change_summary") or {}
        cells = summary.get("cells") or {}
        moved = [
            {
                key: int(component[key])
                for key in ("row_delta", "col_delta")
                if key in component
            }
            for component in summary.get("moved_components", [])
        ]
        return {
            "board_changed": bool(result.get("board_changed")),
            "gameplay_changed": bool(result.get("gameplay_changed")),
            "moved": bool(moved),
            "deltas": moved[:4],
            "level_completed": bool(result.get("level_completed")),
            "game_over": bool(result.get("game_over")),
            "changed_cells": int(cells.get("count", 0)),
        }

    @staticmethod
    def _matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
        for key in ("board_changed", "gameplay_changed", "moved", "level_completed", "game_over"):
            if key in expected and observed[key] != expected[key]:
                return False
        requested_delta = {
            key: expected[key] for key in ("row_delta", "col_delta") if key in expected
        }
        if requested_delta and not any(
            all(delta.get(key) == value for key, value in requested_delta.items())
            for delta in observed["deltas"]
        ):
            return False
        return True

    def observe(
        self, action: str, expected: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        observed = self._observed_effect(result)
        matched = self._matches(expected, observed)
        record = {
            "action": action, "expected": expected, "matched": matched, "observed": observed,
        }
        self.recent.append(record)
        while len(self.recent) > self.max_recent:
            self.recent.popleft()
        key = (action, json.dumps(expected, sort_keys=True, separators=(",", ":")))
        stats = self.rule_stats.setdefault(
            key, {"action": action, "expected": expected, "confirmed": 0, "mismatches": 0},
        )
        stats["confirmed" if matched else "mismatches"] += 1
        stats["latest_observed"] = observed
        if not matched:
            self.replan_required = record
        return record

    def acknowledge_replan(self, note: str) -> dict[str, Any]:
        mismatch = self.replan_required
        if mismatch is None:
            return {"status": "no_replan_was_required"}
        self.replan_required = None
        return {
            "status": "replan_acknowledged",
            "note": note,
            "mismatch": mismatch,
        }

    def needs_replan(self) -> bool:
        return self.replan_required is not None

    def snapshot(self) -> dict[str, Any]:
        rules = sorted(
            self.rule_stats.values(),
            key=lambda rule: (-int(rule["confirmed"]), int(rule["mismatches"]), rule["action"]),
        )
        return {
            "repeat_prediction_required": True,
            "replan_required": self.needs_replan(),
            "latest_mismatch": self.replan_required,
            "verified_rules": [rule for rule in rules if rule["confirmed"]][:self.max_recent],
            "recent_checks": list(self.recent),
            "note": (
                "A previously observed non-RESET action requires predict(action, expected) "
                "in the same Python call. A mismatch means re-plan from the observed state."
            ),
        }



class CodeLimitExceeded(RuntimeError):
    pass


@dataclass
class WorldModel:
    """Small, durable evidence store for hypotheses across context eviction."""

    max_entries: int = field(default_factory=lambda: _env_int("MY_AGENT3_WORLD_NOTES", 10))
    entries: list[dict[str, Any]] = field(default_factory=list)

    def remember(
        self, claim: Any, status: Any, evidence: Any, level: int | None,
    ) -> dict[str, Any]:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        if status not in {"verified", "hypothesis", "rejected"}:
            raise ValueError("status must be verified, hypothesis, or rejected")
        if not isinstance(evidence, str):
            raise TypeError("evidence must be a string")
        entry = {
            "claim": claim.strip()[:240], "status": status,
            "evidence": evidence.strip()[:320], "level": level,
        }
        self.entries = [
            item for item in self.entries
            if not (item["claim"] == entry["claim"] and item["level"] == level)
        ]
        self.entries.append(entry)
        self.entries = self.entries[-self.max_entries:]
        return dict(entry)

    def snapshot(self, level: int | None) -> dict[str, Any]:
        return {
            "current_level": [item for item in self.entries if item["level"] == level],
            "prior_verified": [
                item for item in self.entries
                if item["level"] != level and item["status"] == "verified"
            ][-3:],
            "note": "Prior-level rules are candidates to re-check, not guaranteed to transfer.",
        }


class PythonToolRuntime:
    """Per-game state injected into each ephemeral Python tool call."""

    MAX_CODE_CHARS = 16_000
    MAX_TRACE_EVENTS = 20_000
    MAX_OUTPUT_CHARS = 6_000

    def __init__(self, game: taaf.game.Game, max_actions: int, logger: RunLogger) -> None:
        self.game = game
        self.logger = logger
        self.max_actions = max_actions
        self.actions_taken = 0
        self.history: list[TransitionView] = []
        self.motion_hypotheses = MotionHypothesisTracker()
        self.edge_signals = EdgeSignalTracker()
        self.exploration_ledger = ExplorationLedger()
        self.prediction_ledger = PredictionLedger()
        self.world_model = WorldModel()
        self.previous_frame: FrameView | None = None
        self.current_frame = self._frame_view()
        self.exploration_ledger.visit(self.current_frame, self.edge_signals.ignored_regions())
        self.last_action: str | None = None
        self.last_action_result: dict[str, Any] = {}
        self.last_error: str | None = None
        self.logger.board("initial_board", self.current_frame)

    def _frame_view(self, board: np.ndarray | None = None) -> FrameView:
        state = self.game.current_state
        if board is None:
            board = _board_from_state(state)
        raw = getattr(state, "raw", None)
        return FrameView(
            ascii=_ascii(board), segmentation=_components(board), shape=tuple(board.shape),
            step=getattr(raw, "step", None), level=getattr(raw, "levels_completed", None),
        )

    def valid_actions(self) -> list[str]:
        return [_action_name(value) for value in self.game.current_state.available_actions]

    def terminal(self) -> bool:
        run = self.game.game_run
        if run is None or run.state != "playing":
            return True
        state = getattr(self.game.current_state.raw, "state", None)
        return state == getattr(arcengine.GameState, "WIN", None)

    def game_over(self) -> bool:
        return getattr(self.game.current_state.raw, "state", None) == getattr(
            arcengine.GameState, "GAME_OVER", None
        )

    def _parse_action(self, request: Any) -> tuple[arcengine.GameAction, dict[str, Any]]:
        if isinstance(request, dict):
            raw = request.get("action", request.get("name", request.get("id")))
            params = dict(request.get("data") or {})
            # Duck-style mouse calls are row/col; ARC Engine takes x/y.
            if "row" in request:
                params.setdefault("y", request["row"])
            if "col" in request:
                params.setdefault("x", request["col"])
            for key in ("x", "y"):
                if key in request:
                    params.setdefault(key, request[key])
        else:
            raw, params = request, {}
        action = _game_action(raw)
        if action.name not in self.valid_actions():
            raise ValueError(f"{action.name} is not currently legal; valid_actions={self.valid_actions()}")
        if action.name == "ACTION6":
            if not {"x", "y"}.issubset(params):
                raise ValueError(
                    "ACTION6 is a click and requires x/y or row/col; "
                    "use action([{'action': 'ACTION6', 'row': 4, 'col': 7}])"
                )
            height, width = self.current_frame.shape
            for key, limit in (("x", width), ("y", height)):
                value = params[key]
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise TypeError(f"ACTION6 {key} must be an integer")
                if not 0 <= value < limit:
                    raise ValueError(f"ACTION6 {key}={value} is outside the board")
                params[key] = int(value)
        return action, params

    def view_region(self, row: int, col: int, height: int, width: int) -> dict[str, Any]:
        """Return a strict coordinate-labelled crop for visual motif inspection."""
        return _region_view(_board_from_state(self.game.current_state), row, col, height, width)

    def view_region_clamped(self, row: int, col: int, height: int, width: int) -> dict[str, Any]:
        """Return a crop clipped to the board and configured cell limit."""
        return _clamped_region_view(
            _board_from_state(self.game.current_state), row, col, height, width
        )

    def remember(self, claim: Any, status: Any, evidence: Any = "") -> dict[str, Any]:
        """Store a concise, level-scoped world-model claim across model turns."""
        entry = self.world_model.remember(claim, status, evidence, self.current_frame.level)
        self.logger.record("world_model_note", **entry)
        return entry

    def view_component(self, component_id: int, padding: int = 1) -> dict[str, Any]:
        """Return a bounded crop around one current-frame component."""
        if isinstance(component_id, bool) or not isinstance(component_id, int):
            raise TypeError("component_id must be an integer")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        node = next(
            (item for item in self.current_frame.segmentation["nodes"] if item["id"] == component_id),
            None,
        )
        if node is None:
            raise ValueError(f"component_id={component_id} is not in the current frame")
        row1, col1, row2, col2 = node["bbox"]
        board = _board_from_state(self.game.current_state)
        top, left = max(0, row1 - padding), max(0, col1 - padding)
        bottom, right = min(board.shape[0] - 1, row2 + padding), min(board.shape[1] - 1, col2 + padding)
        return {
            "component": _component_brief(node),
            **_clamped_region_view(board, top, left, bottom - top + 1, right - left + 1),
        }

    def compare_regions(
        self,
        row1: int,
        col1: int,
        row2: int,
        col2: int,
        height: int,
        width: int,
        rotations: bool = True,
        reflections: bool = True,
    ) -> dict[str, Any]:
        """Compare same-sized visual motifs under rotation/reflection."""
        return _compare_regions(
            _board_from_state(self.game.current_state), row1, col1, row2, col2,
            height, width, rotations, reflections,
        )

    def replan(self, note: Any) -> dict[str, Any]:
        """Acknowledge and describe the investigation after a failed prediction."""
        if not isinstance(note, str) or not note.strip():
            raise ValueError("replan note must be a non-empty string")
        acknowledgement = self.prediction_ledger.acknowledge_replan(note.strip())
        self.logger.record(
            "replan_acknowledged", status=acknowledgement["status"], note=note.strip()
        )
        return acknowledgement

    def predict(self, action_name: Any, expected: Any) -> dict[str, Any]:
        """Register a check for the next same-call action without changing the board."""
        action = _game_action(action_name)
        if action.name not in self.valid_actions():
            raise ValueError(f"{action.name} is not currently legal; valid_actions={self.valid_actions()}")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("expected must be a non-empty dict")
        allowed = {
            "board_changed", "gameplay_changed", "moved", "row_delta", "col_delta",
            "level_completed", "game_over",
        }
        unknown = sorted(set(expected) - allowed)
        if unknown:
            raise ValueError(f"unsupported prediction fields: {unknown}")
        normalized: dict[str, Any] = {}
        for key, value in expected.items():
            if key in {"board_changed", "gameplay_changed", "moved", "level_completed", "game_over"}:
                if not isinstance(value, bool):
                    raise TypeError(f"prediction {key} must be a bool")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"prediction {key} must be an integer")
            normalized[key] = value
        return self.prediction_ledger.set_pending(action.name, normalized)



    def action(self, requests: Any) -> list[dict[str, Any]]:
        """Execute one or more legal actions, returning compact transition metadata."""
        if not isinstance(requests, list):
            requests = [requests]
        # Re-grounding after one move is much safer while discovering an
        # unknown game.  A caller may explicitly raise this cap only after it
        # has verified a reliable short sequence.
        batch_cap = _env_int("MY_AGENT3_MAX_BATCH_ACTIONS", 1)
        if len(requests) > batch_cap:
            raise ValueError(f"A Python call may execute at most {batch_cap} actions")
        results = []
        for request in requests:
            if self.actions_taken >= self.max_actions:
                raise RuntimeError("Per-game action budget exhausted")
            if self.terminal():
                break
            before = self.current_frame
            before_board = _board_from_state(self.game.current_state).copy()
            action, data = self._parse_action(request)
            if action.name != "RESET" and self.prediction_ledger.needs_replan():
                self.logger.record("replan_required", action=action.name)
                raise ValueError(
                    "the previous prediction mismatched; inspect the changed region and call "
                    "replan('what the mismatch means') before another non-RESET action"
                )
            prediction = self.prediction_ledger.take_pending(action.name)
            if action.name != "RESET" and self.exploration_ledger.has_tested_this_level(before, action.name) and prediction is None:
                self.logger.record("prediction_required", action=action.name)
                raise ValueError(
                    f"{action.name} was already observed this level; call "
                    "predict('ACTION_NAME', expected) before repeating it"
                )
            self.logger.record("action_requested", action=action.name, data=data)
            self.game.execute_action(arcengine.ActionInput(id=action, data=data))
            self.actions_taken += 1
            self.previous_frame = before
            after_board = _board_from_state(self.game.current_state)
            self.current_frame = self._frame_view(after_board)
            change_summary = _transition_summary(
                before_board, after_board, before.segmentation, self.current_frame.segmentation,
            )
            self.motion_hypotheses.observe(action.name, change_summary, before.segmentation)
            visible_keys = {
                _motion_member_key(node) for node in before.segmentation["nodes"]
            }
            tracked_entity_visible = any(
                all(member in visible_keys for member in members)
                for members in self.motion_hypotheses.candidates
            )
            if before.level != self.current_frame.level:
                self.edge_signals.candidates.clear()
                hud_assessment = {
                    "classification": "level_changed",
                    "gameplay_changed": True,
                    "candidate_indicators": [],
                }
            else:
                hud_assessment = self.edge_signals.observe(
                    before, self.current_frame, change_summary, tracked_entity_visible
                )
            raw = self.game.current_state.raw
            board_changed = before.ascii != self.current_frame.ascii
            result = {
                "action": action.name,
                "board_changed": board_changed,
                "gameplay_changed": (
                    hud_assessment["gameplay_changed"]
                    or before.level != self.current_frame.level
                ),
                "hud_assessment": hud_assessment,
                "change_summary": change_summary,
                "motion_hypotheses": self.motion_hypotheses.snapshot(),
                "level_completed": before.level != self.current_frame.level,
                "game_over": getattr(raw, "state", None) == getattr(arcengine.GameState, "GAME_OVER", None),
                "done": self.terminal(),
                "valid_actions": self.valid_actions(),
                "actions_taken": self.actions_taken,
            }
            ignored_regions = self.edge_signals.ignored_regions()
            self.exploration_ledger.observe(
                before, action.name, result, self.current_frame, ignored_regions
            )
            if prediction is not None:
                result["prediction_check"] = self.prediction_ledger.observe(action.name, prediction, result)
            result["exploration_ledger"] = self.exploration_ledger.snapshot(
                self.current_frame, self.valid_actions(), ignored_regions
            )
            result["prediction_ledger"] = self.prediction_ledger.snapshot()
            self.last_action, self.last_action_result = action.name, result
            self.history.append(TransitionView(action.name, before, self.current_frame, result))
            self.logger.record("action_result", **result)
            self.logger.board("board_after_action", self.current_frame)
            results.append(result)
            if result["done"] or result["level_completed"]:
                break
        return results

    @staticmethod
    def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        allowed = {"numpy", "hashlib", "bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json", "math", "operator", "random", "re", "statistics", "string"}
        root = name.split(".", 1)[0]
        if level or root not in allowed:
            raise ImportError(f"Import {name!r} is not available in the Python tool")
        return __import__(name, globals_, locals_, fromlist, level)

    def execute(self, code: str) -> str:
        if not isinstance(code, str) or not code.strip():
            return json.dumps({"error": "python.code must be a non-empty string"})
        if len(code) > self.MAX_CODE_CHARS:
            return json.dumps({"error": f"code exceeds {self.MAX_CODE_CHARS} characters"})
        if "__" in code:
            return json.dumps({"error": "dunder names are disabled in the Python tool"})
        if _uses_direct_ascii(code):
            self.logger.record("direct_ascii_blocked")
            return json.dumps({"error": "Direct .ascii access is disabled; use view_region or view_component."})
        self.prediction_ledger.clear_pending()
        output = io.StringIO()
        events = 0

        def trace(frame: Any, event: str, arg: Any) -> Callable[..., Any]:
            # action() enters ARC Engine while the model program is executing.
            # Tracing the engine's render loop incorrectly spent this budget on
            # framework code and raised CodeLimitExceeded inside perform_action.
            if frame.f_code.co_filename != "<arc-python-tool>":
                return None
            nonlocal events
            if event == "line":
                events += 1
                if events > self.MAX_TRACE_EVENTS:
                    raise CodeLimitExceeded("Python instruction budget exceeded")
            return trace

        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "dir": dir, "enumerate": enumerate, "filter": filter, "float": float,
            "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
            "isinstance": isinstance, "type": type,
            "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError, "RuntimeError": RuntimeError,
            "NameError": NameError,
            "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
            "print": print, "range": range, "reversed": reversed, "round": round, "set": set,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
            "__import__": self._safe_import,
        }
        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins, "current_frame": self.current_frame,
            "previous_frame": self.previous_frame, "history": self.history,
            "transitions": self.history, "last_transition": self.history[-1] if self.history else None,
            "last_action": self.last_action, "last_action_result": self.last_action_result,
            "last_error": self.last_error, "np": np,
            "valid_actions": self.valid_actions(),
            "motion_hypotheses": self.motion_hypotheses.snapshot(),
            "exploration_ledger": self.exploration_ledger.snapshot(
                self.current_frame, self.valid_actions(), self.edge_signals.ignored_regions()
            ),
            "prediction_ledger": self.prediction_ledger.snapshot(),
            "world_model": self.world_model.snapshot(self.current_frame.level),
            "hud_signals": self.last_action_result.get("hud_assessment", {}),
            "components": self.current_frame.segmentation["nodes"],
            "view_region": self.view_region,
            "view_region_clamped": self.view_region_clamped,
            "replan": self.replan,
            "remember": self.remember,
            "view_component": self.view_component,
            "compare_regions": self.compare_regions,
            "predict": self.predict,
        }
        action_results: list[dict[str, Any]] = []

        def action_wrapper(requests: Any) -> list[dict[str, Any]]:
            """Refresh global tool variables before Python continues after action()."""
            result = self.action(requests)
            compact_results = [_compact_transition_result(item) for item in result]
            action_results.extend(compact_results)
            namespace.update({
                "current_frame": self.current_frame,
                "previous_frame": self.previous_frame,
                "last_transition": self.history[-1] if self.history else None,
                "last_action": self.last_action,
                "last_action_result": self.last_action_result,
                "valid_actions": self.valid_actions(),
                "motion_hypotheses": self.motion_hypotheses.snapshot(),
                "exploration_ledger": self.exploration_ledger.snapshot(
                    self.current_frame, self.valid_actions(), self.edge_signals.ignored_regions()
                ),
                "prediction_ledger": self.prediction_ledger.snapshot(),
                "world_model": self.world_model.snapshot(self.current_frame.level),
                "hud_signals": self.last_action_result.get("hud_assessment", {}),
                "components": self.current_frame.segmentation["nodes"],
            })
            return compact_results

        namespace["action"] = action_wrapper
        self.logger.record("python_script", code=code)
        try:
            try:
                compiled = compile(code, "<arc-python-tool>", "exec")
            except SyntaxError:
                repaired = _repair_trailing_tool_artifact(code)
                if repaired is None:
                    raise
                compiled = compile(repaired, "<arc-python-tool>", "exec")
                self.logger.record("python_code_repaired", original=code, repaired=repaired)
                output.write("INFO repaired trailing tool-call artifact\n")
            with contextlib.redirect_stdout(output):
                previous_trace = sys.gettrace()
                sys.settrace(trace)
                try:
                    exec(compiled, namespace, namespace)
                finally:
                    sys.settrace(previous_trace)
            if "result" in namespace:
                rendered = json.dumps(namespace["result"], default=str)
                if output.tell():
                    output.write("\n")
                output.write(rendered)
        except Exception as exc:  # Tool errors are feedback for the model, not game failures.
            output.write(f"ERROR {type(exc).__name__}: {exc}")
        # Models naturally write action([...]) as a statement.  Never make
        # them infer that they must assign its return value just to receive the
        # observation needed for their next decision.  Keep the feedback even
        # if later code in the same tool call raises an exception.
        if action_results:
            if output.tell():
                output.write("\n")
            output.write(json.dumps({"action_feedback": action_results}, default=str))
        text = output.getvalue().strip()
        if len(text) > self.MAX_OUTPUT_CHARS:
            text = text[: self.MAX_OUTPUT_CHARS] + "\n[tool output truncated]"
        result = text or "{" + '"ok": true' + "}"
        self.logger.record("python_result", output=result)
        return result


SYSTEM_PROMPT = """You are solving an unknown ARC-AGI-3 game efficiently.

You have exactly one tool, python. Use it to inspect the current state and to
act. Do not return an action in prose: call action(...) inside Python.

Each Python call begins with current_frame, previous_frame, history,
transitions, last_transition, last_action, last_action_result, and valid_actions.
It also begins with motion_hypotheses: conservative action-to-translation
evidence for individual components and co-moving groups. It intentionally
excludes objects that only grow, shrink, appear, or disappear.
exploration_ledger tracks tested actions and revisited gameplay states,
excluding only edge regions verified as steadily changing indicators. It is
evidence, not an instruction to act blindly. prediction_ledger retains checks
for model-stated predictions. predict(action, expected) registers one check for
the next action in the same Python call.
world_model contains compact notes that persist across context eviction; call
remember(claim, status, evidence) to preserve important facts and hypotheses.
Status is verified, hypothesis, or rejected.
current_frame has .segmentation, .shape, .step, and .level; components is a
shorthand for current_frame.segmentation['nodes']. Direct .ascii access is
disabled: inspect focused crops with view_region_clamped or view_component.
Crops return [row_min, col_min, row_max, col_max] plus row/column labels. Rows
increase downward; columns increase rightward. hud_signals reports edge
components whose repeated monotonic changes may indicate HUD; do not treat them
as proof of puzzle progress.
segmentation is {'nodes': [...], 'adjacency_list': [...]}; node ids are local to
the current frame. Use compact summaries rather than printing whole boards.
Optional visual helpers are view_region_clamped(row, col, height, width),
view_region(row, col, height, width), view_component(component_id, padding=1),
and compare_regions(row1, col1, row2, col2, height, width). The clamped view
clips to board bounds and the output-cell limit. compare_regions tests exact and
colour-remapped patterns, including rotations/reflections by default. Before
calling an object a goal, inspect its local pattern and either compare it with a
related motif or obtain an action outcome that supports the claim. Do not infer
a goal from an icon's colour or bounding box alone.

Call action(["ACTION1"]) for a directional action. ACTION6 is a coordinate click:
action([{"action": "ACTION6", "row": 4, "col": 7}])
Always include both coordinates for ACTION6. Use predict("ACTION6", expected)
before a repeated click; predictions take an action name, never coordinates.
MOUSE is accepted as an ACTION6 alias, but prefer the names in valid_actions.
The runtime validates actions and refreshes every state variable immediately.
For a non-RESET action already observed this level, call predict immediately
before action, for example:
predict("ACTION1", {"gameplay_changed": True, "moved": True, "row_delta": -5})
action(["ACTION1"])
Valid expectation keys are board_changed, gameplay_changed, moved, row_delta,
col_delta, level_completed, and game_over. board_changed includes any pixel
change; gameplay_changed discounts only a strongly repeated edge indicator.
Prefer gameplay_changed and relevant object movement when predicting progress.
First probes need no prediction. A mismatch sets
prediction_ledger.replan_required. Inspect the local change or relevant
components, update world_model, then call replan("what the mismatch implies")
before another non-RESET action. Treat the observed result as ground truth.
Every action call automatically returns action_feedback, even if your Python
code does not assign or print its return value. action_feedback is a list of compact results. Each result['change']
reports changed cells and conservatively matched moved components with explicit
row_delta (negative=up) and col_delta (negative=left). Use this feedback
before making another claim. Motion rules vary by location and blockers; a
successful move elsewhere does not predict movement at a boundary. Re-check
current object location and latest non-translation evidence before repeating.
Use exploration_ledger["this_level"] and
exploration_ledger["current_state"]["tested_actions"] to spot tried actions and
gameplay-state revisits; avoid cycles that do not progress.
When unclear, choose one action that distinguishes live hypotheses. Record only
useful evidence with remember(...), and mark unresolved interpretations as
hypothesis. Once the goal and action rules have evidence, search or plan over
the inferred state space instead of continuing exploratory moves. Games may not
have a moving player. After every action, distinguish object/goal changes from
likely HUD changes; board_changed alone does not mean progress.
Do not repeatedly print full boards after relevant evidence is known. Use one
action per Python call while discovering mechanics; batch only a short sequence
whose transitions have already been verified. Available imports include numpy,
math, collections, itertools, hashlib, and json. np is available without import.
Use isinstance, type, hasattr, and try/except Exception normally. last_transition
is an object: use .result and .after_frame, not dictionary subscripting. For
compact action return values use r[0]['change']; full last_action_result uses
'change_summary'. Crops return an ascii string; use splitlines() for rows.
Keep reasoning concise and call
python promptly.
"""


class PythonToolAgent:
    """OpenAI-compatible tool loop with bounded context eviction."""

    def __init__(self, runtime: PythonToolRuntime, analyzer_timeout: float = 120.0) -> None:
        self.runtime = runtime
        timeout = httpx.Timeout(
            connect=float(os.getenv("MY_AGENT3_CONNECT_TIMEOUT", "10")),
            read=float(os.getenv("MY_AGENT3_READ_TIMEOUT", str(analyzer_timeout))),
            write=float(os.getenv("MY_AGENT3_WRITE_TIMEOUT", "120")),
            pool=float(os.getenv("MY_AGENT3_POOL_TIMEOUT", "10")),
        )
        self.client = AsyncOpenAI(
            base_url=os.getenv("LOCAL_ANALYZER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:1234/v1",
            api_key=os.getenv("LOCAL_ANALYZER_API_KEY") or os.getenv("OPENAI_API_KEY") or "local",
            http_client=httpx.AsyncClient(timeout=timeout, trust_env=False),
            max_retries=0,
        )
        self.model = os.getenv("INFERENCE_ANALYZER_MODEL") or os.getenv("LOCAL_ANALYZER_MODEL_ID") or "vrfai/Qwen3.6-27B-FP8"
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_messages = _env_int("MY_AGENT3_MESSAGE_LIMIT", 18, 6)
        self.max_tool_calls = _env_int("MY_AGENT3_TOOL_CALLS_PER_TURN", 6)
        self.max_tokens = _env_int("MY_AGENT3_MAX_TOKENS", 2048)
        self.forced_tool_choice_supported = True
        self.context_tokens = _env_int("MY_AGENT3_CONTEXT_TOKENS", 32768)
        self.context_margin = _env_int("MY_AGENT3_CONTEXT_MARGIN", 512)
        self.tokenizer_client = httpx.AsyncClient(timeout=10.0, trust_env=False)
        self.tokenizer_url = str(self.client.base_url).rstrip("/").removesuffix("/v1") + "/tokenize"
        self.tokenizer_available = True

    async def _prompt_tokens(self) -> int:
        """Count the server's templated multimodal prompt, or estimate conservatively."""
        if self.tokenizer_available:
            try:
                response = await self.tokenizer_client.post(self.tokenizer_url, json={
                    "model": self.model, "messages": self.messages, "tools": [PYTHON_TOOL],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                })
                response.raise_for_status()
                payload = response.json()
                count = int(payload["count"])
                if count < 0:
                    raise ValueError("Token count cannot be negative")
                self.context_tokens = min(self.context_tokens, int(payload["max_model_len"]))
                return count
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                self.tokenizer_available = False
                self.runtime.logger.record("tokenizer_fallback", error=str(exc))
        # Text bytes deliberately overestimate typical BPE text; image tokens
        # depend on the model processor. Server context errors still get a bounded retry.
        estimate = len(json.dumps(PYTHON_TOOL, ensure_ascii=False).encode()) + 256
        for message in self.messages:
            compact = dict(message)
            content = compact.get("content")
            if isinstance(content, list):
                compact["content"] = [part for part in content if part.get("type") != "image_url"]
                estimate += sum(part.get("type") == "image_url" for part in content) * _env_int(
                    "MY_AGENT3_IMAGE_TOKEN_RESERVE", 4096,
                )
            estimate += len(json.dumps(compact, ensure_ascii=False).encode()) + 32
        return estimate

    def _drop_context_group(self) -> int:
        """Drop old state or a complete assistant/tool exchange, keeping the latest state."""
        users = [i for i, message in enumerate(self.messages) if message["role"] == "user"
                 and message.get("content") != "Call the python tool now; do not answer in prose."]
        if len(users) > 1:
            # Drop history before the latest state one complete turn at a time.
            stop = users[1]
            removed = stop - 1
            del self.messages[1:stop]
            return removed
        first = users[0] + 1 if users else 1
        assistants = [i for i in range(first, len(self.messages)) if self.messages[i]["role"] == "assistant"]
        if len(assistants) > 1:
            start, stop = assistants[:2]
            del self.messages[start:stop]
            return stop - start
        return 0

    async def _evict(self, reason: str = "message_limit") -> None:
        """Reserve output space and evict complete tool exchanges before every request."""
        discarded = 0
        prompt_tokens = await self._prompt_tokens()
        budget = self.context_tokens - self.max_tokens - self.context_margin
        while len(self.messages) > self.max_messages or prompt_tokens > budget:
            removed = self._drop_context_group()
            if not removed:
                break
            discarded += removed
            prompt_tokens = await self._prompt_tokens()
        if discarded:
            self.runtime.logger.record(
                "context_evicted", reason=reason, discarded_messages=discarded,
                retained_messages=len(self.messages), prompt_tokens=prompt_tokens,
                input_budget=budget,
            )

    async def retain_after_inspection(self) -> None:
        """Retry without discarding recent inspection, reasoning, or tool feedback."""
        await self._evict(reason="inspection_retry")
        self.runtime.logger.record(
            "context_retained", reason="inspection_retry", message_count=len(self.messages),
        )

    def _turn_message(self) -> dict[str, Any]:
        frame = self.runtime.current_frame
        summary = {
            "shape": frame.shape, "level": frame.level, "step": frame.step,
            "valid_actions": self.runtime.valid_actions(),
            "last_action": self.runtime.last_action,
            "last_transition": _compact_transition_result(self.runtime.last_action_result)
            if self.runtime.last_action_result else None,
            "motion_hypotheses": self.runtime.motion_hypotheses.snapshot(),
            "exploration_ledger": self.runtime.exploration_ledger.snapshot(
                frame, self.runtime.valid_actions(), self.runtime.edge_signals.ignored_regions()
            ),
            "hud_signals": self.runtime.last_action_result.get("hud_assessment", {}),
            "world_model": self.runtime.world_model.snapshot(frame.level),
            "last_error": self.runtime.last_error,
            "prediction_ledger": self.runtime.prediction_ledger.snapshot(),
            "components": [
                {key: node[key] for key in ("id", "color", "pixels", "bbox", "hash", "children")}
                for node in frame.segmentation["nodes"][:100]
            ],
        }
        board = _board_from_state(self.runtime.game.current_state)
        return {"role": "user", "content": [
            {"type": "text", "text": "Current state:\n" + json.dumps(summary)},
            {"type": "image_url", "image_url": {"url": _image_url(board)}},
        ]}

    async def close(self) -> None:
        await self.tokenizer_client.aclose()
        await self.client.close()

    async def preflight(self) -> bool:
        """Verify the multimodal tool protocol without changing game state."""
        timeout = _env_float("MY_AGENT3_PREFLIGHT_TIMEOUT", 45.0, 1.0)
        max_tokens = _env_int("MY_AGENT3_PREFLIGHT_MAX_TOKENS", self.max_tokens)
        messages = [
            self.messages[0],
            self._turn_message(),
            {
                "role": "user",
                "content": (
                    "Preflight check: call the python tool now with exactly "
                    "print('preflight-ok'). Do not analyze the board or call action()."
                ),
            },
        ]
        started = time.monotonic()
        self.runtime.logger.record(
            "model_preflight_request", model=self.model, timeout_s=timeout, max_tokens=max_tokens,
            valid_actions=self.runtime.valid_actions(),
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=[PYTHON_TOOL], tool_choice="required",
                temperature=0, max_tokens=max_tokens, timeout=timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
        except Exception as exc:
            self.runtime.last_error = f"preflight {type(exc).__name__}: {exc}"
            self.runtime.logger.record(
                "model_preflight_error", error=self.runtime.last_error,
                duration_s=round(time.monotonic() - started, 3),
            )
            return False

        choices = list(getattr(response, "choices", []) or [])
        choice = choices[0] if choices else None
        message = choice.message if choice else None
        calls = list(getattr(message, "tool_calls", None) or [])
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        finish_reason = getattr(choice, "finish_reason", None)
        truncated = finish_reason in {"length", "max_tokens"} or (
            isinstance(completion_tokens, int) and completion_tokens >= max_tokens
        )
        valid_call = False
        for call in calls:
            if call.function.name != "python":
                continue
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(arguments.get("code"), str) and arguments["code"].strip():
                valid_call = True
                break
        self.runtime.logger.record(
            "model_preflight_response", duration_s=round(time.monotonic() - started, 3),
            valid_tool_call=valid_call,
            truncated=truncated,
            finish_reason=finish_reason,
            content=getattr(message, "content", None),
            reasoning=(
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
            ),
            tool_calls=[call.model_dump() for call in calls],
            tool_arguments=[call.function.arguments for call in calls],
            usage=usage.model_dump() if usage else None,
        )
        if not valid_call:
            if truncated:
                self.runtime.last_error = (
                    f"preflight_truncated: response reached the {max_tokens}-token limit "
                    "before a valid python tool call"
                )
                failure_kind = "preflight_truncated"
            else:
                self.runtime.last_error = "preflight did not return a valid python tool call"
                failure_kind = "invalid_tool_call"
            self.runtime.logger.record(
                "model_preflight_error", error=self.runtime.last_error,
                failure_kind=failure_kind, finish_reason=finish_reason,
            )
        return valid_call

    async def play_turn(self) -> bool:
        """Return whether a real environment action was executed."""
        before = self.runtime.actions_taken
        self.messages.append(self._turn_message())
        self.runtime.logger.record(
            "model_request",
            model=self.model,
            valid_actions=self.runtime.valid_actions(),
            level=self.runtime.current_frame.level,
            step=self.runtime.current_frame.step,
            message_count=len(self.messages),
        )
        await self._evict()
        context_retries = 0
        tool_attempts = 0
        while tool_attempts < self.max_tool_calls:
            await self._evict(reason="request_budget")
            request_started = time.monotonic()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model, messages=self.messages, tools=[PYTHON_TOOL],
                    tool_choice="required" if self.forced_tool_choice_supported else "auto",
                    temperature=0, max_tokens=self.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
            except BadRequestError as exc:
                detail = str(exc).lower()
                if any(term in detail for term in ("maximum context length", "context_length_exceeded", "max model len")):
                    if context_retries < 3 and self._drop_context_group():
                        context_retries += 1
                        self.runtime.logger.record(
                            "context_retry", retry=context_retries, error=str(exc),
                            retained_messages=len(self.messages),
                        )
                        continue
                if self.forced_tool_choice_supported and (
                    "tool_choice" in detail or "tool choice" in detail
                ) and any(word in detail for word in ("unsupported", "not supported", "only", "must be")):
                    self.forced_tool_choice_supported = False
                    self.runtime.logger.record(
                        "tool_choice_fallback", error=str(exc),
                        duration_s=round(time.monotonic() - request_started, 3),
                    )
                    continue
                self.runtime.last_error = f"{type(exc).__name__}: {exc}"
                self.runtime.logger.record(
                    "model_error", error=self.runtime.last_error,
                    duration_s=round(time.monotonic() - request_started, 3),
                )
                break
            except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                self.runtime.last_error = f"{type(exc).__name__}: {exc}"
                self.runtime.logger.record(
                    "model_error", error=self.runtime.last_error,
                    duration_s=round(time.monotonic() - request_started, 3),
                )
                break
            tool_attempts += 1
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            self.runtime.logger.record(
                "model_response",
                finish_reason=response.choices[0].finish_reason,
                content=message.content,
                reasoning=(
                    getattr(message, "reasoning", None)
                    or getattr(message, "reasoning_content", None)
                ),
                tool_calls=[call.model_dump() for call in calls],
                usage=response.usage.model_dump() if response.usage else None,
                duration_s=round(time.monotonic() - request_started, 3),
            )
            dumped: dict[str, Any] = {"role": "assistant", "tool_calls": [call.model_dump() for call in calls]}
            if message.content:
                dumped["content"] = message.content
            self.messages.append(dumped)
            if not calls:
                self.messages.append({"role": "user", "content": "Call the python tool now; do not answer in prose."})
                continue
            action_executed = False
            for call in calls:
                if action_executed:
                    # vLLM may emit several tool calls in one response.  Leave
                    # their game-changing work for a fresh, fully grounded turn
                    # while still satisfying the response protocol for each id.
                    result = json.dumps({
                        "deferred": True,
                        "reason": "An earlier tool call changed the game state; re-grounding now.",
                    })
                elif call.function.name != "python":
                    result = json.dumps({"error": "python is the only available tool"})
                else:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                        result = self.runtime.execute(arguments.get("code"))
                    except (TypeError, json.JSONDecodeError) as exc:
                        result = json.dumps({"error": f"invalid python arguments: {exc}"})
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if (
                    self.runtime.terminal()
                    or self.runtime.game_over()
                    or self.runtime.actions_taken >= self.runtime.max_actions
                ):
                    await self._evict()
                    return self.runtime.actions_taken > before
                if self.runtime.actions_taken > before:
                    action_executed = True
            if action_executed:
                await self._evict(reason="action")
                self.runtime.logger.record(
                    "context_retained", reason="action", message_count=len(self.messages),
                )
                return True
            await self._evict()
        return self.runtime.actions_taken > before


@dataclass
class MyAgent3Solver(Solver):
    """TAAF entry point. Set the Kaggle solver to ``MyAgent3Solver``."""

    label: str = "MyAgent3"
    max_actions_per_game: int = 100
    _preflight_ok: bool | None = field(default=None, init=False, repr=False)
    _model_failures: list[dict[str, str]] = field(default_factory=list, init=False, repr=False)

    def _record_model_failure(self, game: taaf.game.Game, error: str) -> None:
        self._model_failures.append({
            "game_id": str(getattr(game, "game_id", None) or getattr(game, "env_name", "unknown")),
            "error": error,
        })

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._model_failures.clear()
        # Only HTTP awaits overlap. Python tools and engine actions stay on the
        # event-loop thread: redirect_stdout and mutable action enums are not thread-safe.
        self._preflight_ok = None
        self._preflight_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))

        async def play(game: taaf.game.Game) -> None:
            async with semaphore:
                await self._play_one(game)

        tasks = [asyncio.create_task(play(game)) for game in games]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _play_until_done(
        self, runtime: PythonToolRuntime, agent: PythonToolAgent, logger: RunLogger,
    ) -> None:
        async with self._preflight_lock:
            if self._preflight_ok is None:
                self._preflight_ok = await agent.preflight()
        if not self._preflight_ok:
            self._record_model_failure(runtime.game, runtime.last_error or "model preflight failed")
            logger.record("solver_stopped", reason="model_preflight_failed", error=runtime.last_error)
            return
        max_no_action_retries = _env_int("MY_AGENT3_MAX_NO_ACTION_RETRIES", 2)
        no_action_retries = 0
        while not runtime.terminal() and runtime.actions_taken < self.max_actions_per_game:
            if runtime.game_over():
                if "RESET" not in runtime.valid_actions():
                    break
                runtime.action(["RESET"])
                continue
            did_act = await agent.play_turn()
            if did_act:
                no_action_retries = 0
                continue
            if runtime.last_error:
                self._record_model_failure(runtime.game, runtime.last_error)
                logger.record(
                    "solver_stopped", reason="model_error", error=runtime.last_error,
                    actions_taken=runtime.actions_taken,
                )
                break
            if no_action_retries < max_no_action_retries:
                no_action_retries += 1
                logger.record(
                    "solver_retrying", reason="inspection_only_turn",
                    retry=no_action_retries, max_retries=max_no_action_retries,
                    actions_taken=runtime.actions_taken,
                )
                await agent.retain_after_inspection()
                continue
            logger.record(
                "solver_stopped", reason="inspection_retry_exhausted",
                actions_taken=runtime.actions_taken, retries=no_action_retries,
            )
            break

    async def _play_one(self, game: taaf.game.Game) -> None:
        logger = RunLogger(game)
        runtime = PythonToolRuntime(game, self.max_actions_per_game, logger)
        agent = PythonToolAgent(runtime, analyzer_timeout=self.analyzer_timeout)
        budget = float(self.max_runtime_s_per_game)
        try:
            async with asyncio.timeout(budget if budget > 0 else None):
                await self._play_until_done(runtime, agent, logger)
        except TimeoutError:
            logger.record(
                "solver_stopped", reason="runtime_budget_exhausted",
                actions_taken=runtime.actions_taken, budget_s=budget,
            )
        except asyncio.CancelledError:
            logger.record("solver_stopped", reason="cancelled", actions_taken=runtime.actions_taken)
            raise
        except Exception as exc:
            print(f"[MYAGENT3 ERROR] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            self._record_model_failure(runtime.game, f"{type(exc).__name__}: {exc}")
            if game.game_run is not None:
                game.game_run.solver_note = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if game.game_run is not None and game.game_run.final_score is None:
                    game.finish_game()
            finally:
                try:
                    await agent.close()
                finally:
                    logger.close()


# Keep the name used by the older Kaggle cell available when switching files.
MyAgentSolver = MyAgent3Solver
