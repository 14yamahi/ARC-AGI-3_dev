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
from collections import deque
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
    BadRequestError,
    OpenAI,
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
            print(f"[MYAGENT3 {event}] {preview}", flush=True)

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


@dataclass
class FrameView:
    ascii: str
    segmentation: dict[str, Any]
    shape: tuple[int, int]
    step: int | None
    level: int | None


@dataclass
class TransitionView:
    action: str
    before_frame: FrameView
    after_frame: FrameView
    result: dict[str, Any]

    @property
    def frame(self) -> FrameView:
        return self.after_frame


class CodeLimitExceeded(RuntimeError):
    pass


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
        self.previous_frame: FrameView | None = None
        self.current_frame = self._frame_view()
        self.last_action: str | None = None
        self.last_action_result: dict[str, Any] = {}
        self.last_error: str | None = None
        self.logger.board("initial_board", self.current_frame)

    def _frame_view(self) -> FrameView:
        state = self.game.current_state
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
                params.setdefault("y", int(request["row"]))
            if "col" in request:
                params.setdefault("x", int(request["col"]))
            for key in ("x", "y"):
                if key in request:
                    params.setdefault(key, int(request[key]))
        else:
            raw, params = request, {}
        action = _game_action(raw)
        if action.name not in self.valid_actions():
            raise ValueError(f"{action.name} is not currently legal; valid_actions={self.valid_actions()}")
        if action.name == "MOUSE" and not {"x", "y"}.issubset(params):
            raise ValueError("MOUSE requires both x and y (or row and col)")
        return action, params

    def action(self, requests: Any) -> list[dict[str, Any]]:
        """Execute one or more legal actions, returning compact transition metadata."""
        if not isinstance(requests, list):
            requests = [requests]
        batch_cap = _env_int("MY_AGENT3_MAX_BATCH_ACTIONS", 8)
        if len(requests) > batch_cap:
            raise ValueError(f"A Python call may execute at most {batch_cap} actions")
        results = []
        for request in requests:
            if self.actions_taken >= self.max_actions:
                raise RuntimeError("Per-game action budget exhausted")
            if self.terminal():
                break
            before = self.current_frame
            action, data = self._parse_action(request)
            self.logger.record("action_requested", action=action.name, data=data)
            self.game.execute_action(arcengine.ActionInput(id=action, data=data))
            self.actions_taken += 1
            self.previous_frame = before
            self.current_frame = self._frame_view()
            raw = self.game.current_state.raw
            result = {
                "action": action.name,
                "board_changed": before.ascii != self.current_frame.ascii,
                "level_completed": before.level != self.current_frame.level,
                "game_over": getattr(raw, "state", None) == getattr(arcengine.GameState, "GAME_OVER", None),
                "done": self.terminal(),
                "valid_actions": self.valid_actions(),
                "actions_taken": self.actions_taken,
            }
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
        allowed = {"bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json", "math", "operator", "random", "re", "statistics", "string"}
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
            "enumerate": enumerate, "filter": filter, "float": float, "frozenset": frozenset,
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
            "valid_actions": self.valid_actions(),
        }

        def action_wrapper(requests: Any) -> list[dict[str, Any]]:
            """Refresh global tool variables before Python continues after action()."""
            result = self.action(requests)
            namespace.update({
                "current_frame": self.current_frame,
                "previous_frame": self.previous_frame,
                "last_transition": self.history[-1] if self.history else None,
                "last_action": self.last_action,
                "last_action_result": self.last_action_result,
                "valid_actions": self.valid_actions(),
            })
            return result

        namespace["action"] = action_wrapper
        self.logger.record("python_script", code=code)
        try:
            with contextlib.redirect_stdout(output):
                previous_trace = sys.gettrace()
                sys.settrace(trace)
                try:
                    exec(compile(code, "<arc-python-tool>", "exec"), namespace, namespace)
                finally:
                    sys.settrace(previous_trace)
            if "result" in namespace:
                rendered = json.dumps(namespace["result"], default=str)
                if output.tell():
                    output.write("\n")
                output.write(rendered)
        except Exception as exc:  # Tool errors are feedback for the model, not game failures.
            output.write(f"ERROR {type(exc).__name__}: {exc}")
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
current_frame has .ascii, .segmentation, .shape, .step, and .level.
segmentation is {'nodes': [...], 'adjacency_list': [...]}; node ids are local to
the current frame. Use compact summaries rather than printing whole boards.

action accepts a list such as action(['ACTION1']) or
action([{'action': 'MOUSE', 'row': 4, 'col': 7}]). It validates actions against
valid_actions, executes them immediately, and refreshes every state variable.
You may call action multiple times or use a short, reliable batch. Stop acting
when a result says done, game_over, or level_completed; re-ground on the next
turn. Explore with discriminating probes, then write BFS/search code when the
mechanism is understood. A game need not have a moving player.
"""


class PythonToolAgent:
    """OpenAI-compatible tool loop with bounded context eviction."""

    def __init__(self, runtime: PythonToolRuntime) -> None:
        self.runtime = runtime
        timeout = httpx.Timeout(
            connect=float(os.getenv("MY_AGENT3_CONNECT_TIMEOUT", "10")),
            read=float(os.getenv("MY_AGENT3_READ_TIMEOUT", "120")),
            write=float(os.getenv("MY_AGENT3_WRITE_TIMEOUT", "120")),
            pool=float(os.getenv("MY_AGENT3_POOL_TIMEOUT", "10")),
        )
        self.client = OpenAI(
            base_url=os.getenv("LOCAL_ANALYZER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:1234/v1",
            api_key=os.getenv("LOCAL_ANALYZER_API_KEY") or os.getenv("OPENAI_API_KEY") or "local",
            http_client=httpx.Client(timeout=timeout, trust_env=False),
            max_retries=0,
        )
        self.model = os.getenv("INFERENCE_ANALYZER_MODEL") or os.getenv("LOCAL_ANALYZER_MODEL_ID") or "vrfai/Qwen3.6-27B-FP8"
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_messages = _env_int("MY_AGENT3_MESSAGE_LIMIT", 18, 6)
        self.max_tool_calls = _env_int("MY_AGENT3_TOOL_CALLS_PER_TURN", 6)
        self.forced_tool_choice_supported = True

    def _evict(self) -> None:
        # Preserve system instructions and the latest complete interaction.
        if len(self.messages) > self.max_messages:
            self.messages = [self.messages[0], *self.messages[-(self.max_messages - 1):]]

    def _turn_message(self) -> dict[str, Any]:
        frame = self.runtime.current_frame
        summary = {
            "shape": frame.shape, "level": frame.level, "step": frame.step,
            "valid_actions": self.runtime.valid_actions(),
            "last_action": self.runtime.last_action,
            "last_action_result": self.runtime.last_action_result,
            "last_error": self.runtime.last_error,
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

    def close(self) -> None:
        self.client.close()

    def preflight(self) -> bool:
        """Verify the exact multimodal, required-tool request before playing."""
        timeout = _env_float("MY_AGENT3_PREFLIGHT_TIMEOUT", 45.0, 1.0)
        messages = [self.messages[0], self._turn_message()]
        started = time.monotonic()
        self.runtime.logger.record(
            "model_preflight_request", model=self.model, timeout_s=timeout,
            valid_actions=self.runtime.valid_actions(),
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=[PYTHON_TOOL], tool_choice="required",
                temperature=0, max_tokens=256, timeout=timeout,
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
        message = choices[0].message if choices else None
        calls = list(getattr(message, "tool_calls", None) or [])
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
            tool_calls=[call.model_dump() for call in calls],
            usage=response.usage.model_dump() if response.usage else None,
        )
        if not valid_call:
            self.runtime.last_error = "preflight did not return a valid python tool call"
            self.runtime.logger.record("model_preflight_error", error=self.runtime.last_error)
        return valid_call

    def play_turn(self) -> bool:
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
        self._evict()
        for _ in range(self.max_tool_calls):
            request_started = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=self.messages, tools=[PYTHON_TOOL],
                    tool_choice="required" if self.forced_tool_choice_supported else "auto",
                    temperature=0, max_tokens=2048,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
            except BadRequestError as exc:
                detail = str(exc).lower()
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
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            self.runtime.logger.record(
                "model_response",
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
            for call in calls:
                if call.function.name != "python":
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
                    self._evict()
                    return self.runtime.actions_taken > before
            self._evict()
        return self.runtime.actions_taken > before


@dataclass
class MyAgent3Solver(Solver):
    """TAAF entry point. Set the Kaggle solver to ``MyAgent3Solver``."""

    label: str = "MyAgent3"
    max_actions_per_game: int = 10
    _preflight_ok: bool | None = field(default=None, init=False, repr=False)

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        # Inference is synchronous. Running games sequentially prevents one
        # vLLM request from blocking every supposedly concurrent coroutine.
        for game in games:
            await self._play_one(game)

    async def _play_one(self, game: taaf.game.Game) -> None:
        logger = RunLogger(game)
        runtime = PythonToolRuntime(game, self.max_actions_per_game, logger)
        agent = PythonToolAgent(runtime)
        try:
            if self._preflight_ok is None:
                self._preflight_ok = agent.preflight()
            if not self._preflight_ok:
                logger.record("solver_stopped", reason="model_preflight_failed", error=runtime.last_error)
                return
            while not runtime.terminal() and runtime.actions_taken < self.max_actions_per_game:
                await asyncio.sleep(0)
                if runtime.game_over():
                    if "RESET" not in runtime.valid_actions():
                        break
                    runtime.action(["RESET"])
                    continue
                did_act = agent.play_turn()
                if did_act:
                    continue
                # Never guess an action after an inference failure.  In
                # particular, MOUSE is legal in some games but requires x/y;
                # an unparameterized fallback previously crashed those games.
                logger.record(
                    "solver_stopped", reason="model_did_not_act", error=runtime.last_error,
                    actions_taken=runtime.actions_taken,
                )
                break
            if game.game_run is not None and game.game_run.final_score is None:
                game.finish_game()
        except asyncio.CancelledError:
            if game.game_run is not None and game.game_run.final_score is None:
                game.finish_game()
            raise
        except Exception as exc:
            print(f"[MYAGENT3 ERROR] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            if game.game_run is not None:
                game.game_run.solver_note = f"{type(exc).__name__}: {exc}"
                if game.game_run.final_score is None:
                    with contextlib.suppress(Exception):
                        game.finish_game()
        finally:
            agent.close()
            logger.close()


# Keep the name used by the older Kaggle cell available when switching files.
MyAgentSolver = MyAgent3Solver
