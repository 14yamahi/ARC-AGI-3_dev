import os
from dataclasses import dataclass
from collections import deque, defaultdict
import numpy as np
from typing import Any
import base64
import io
import json
from openai import OpenAI, BadRequestError
from PIL import Image
import traceback
import contextlib

# used for local runs w/ gpt API
# from agents.agent import Agent
# from arcengine import FrameData, GameAction, GameState

# used for Kaggle notebook runs
import asyncio
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Any
import arcengine
import numpy as np
import taaf.game
from taaf.solver import Solver
import httpx
import hashlib

# Should be already called in Kaggle 
# LOCAL_BASE_URL = os.getenv(
#     "ARC_LOCAL_BASE_URL",
#     "http://127.0.0.1:1234/v1",
# )

# LOCAL_MODEL = os.getenv(
#     "ARC_LOCAL_MODEL",
#     "vrfai/Qwen3.6-27B-FP8",
# )

@dataclass
class GameObject:
    id: int
    color: int
    pixels: set[tuple[int, int]]

    bbox: tuple[int, int, int, int]
    center: tuple[float, float]

    shape_hash: str
    boundary: list[tuple[int, int]]
    children: list[int]
    track_id: str | None = None
    tracking_confidence: float = 0.0
    tracking_status: str = "untracked"
    parent_track_ids: tuple[str, ...] = ()

@dataclass
class Segmentation:
    objects: list[GameObject]
    adjacency: list[tuple[int, int]]

@dataclass
class Movement:
    object_id: int
    before: GameObject
    after: GameObject
    delta: tuple[int, int]

@dataclass
class Transition:
    action: str
    moved_objects: list[Movement]
    appeared_objects: list
    disappeared_objects: list
    changed_objects: list
    global_changes: dict


@dataclass
class InteractionObservation:
    goal_type: str
    target_tracks: list[str]
    action: str
    contacted: bool
    target_not_visible: list[str]
    ui_changes: list[dict]
    gameplay_changes: list[dict]
    association_confidence: str

@dataclass
class SceneModel:
    controlled_entity: set[int]

    likely_walls: list[int]
    likely_floor_colors: set[int]

    interesting_objects: list[int]

    spatial_relations: list

    goal_hypotheses: list

@dataclass
class ActionPrediction:
    action: arcengine.GameAction

    expected_delta: tuple
    collision_probability: float

    affected_objects: list

    goal_progress: float
    risk: float

@dataclass
class SpatialRelation:
    relation: str
    a: int
    b: int

@dataclass
class CompositeRegion:
    container_id: int
    content_ids: list[int]

LOCAL_BASE_URL = (
    os.getenv("LOCAL_ANALYZER_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "http://127.0.0.1:1234/v1"
)

LOCAL_MODEL = (
    os.getenv("INFERENCE_ANALYZER_MODEL")
    or os.getenv("LOCAL_ANALYZER_MODEL_ID")
    or "vrfai/Qwen3.6-27B-FP8"
)

LOCAL_API_KEY = (
    os.getenv("LOCAL_ANALYZER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "local"
)

http_client = httpx.Client(
    trust_env=False,
)

print("MY AGENT BASE URL:", LOCAL_BASE_URL)
print("MY AGENT MODEL:", LOCAL_MODEL)

client = OpenAI(
    base_url=LOCAL_BASE_URL,
    api_key=LOCAL_API_KEY,
    http_client=http_client,
)

# convert ARC frame to png
ARC_PALETTE = {
    0:  (255, 255, 255),  # white
    1:  (204, 204, 204),  # light gray
    2:  (153, 153, 153),  # gray
    3:  (102, 102, 102),  # dark gray
    4:  (51, 51, 51),     # very dark gray
    5:  (0, 0, 0),        # black
    6:  (229, 58, 163),   # magenta
    7:  (255, 123, 204),  # pink
    8:  (249, 60, 49),    # red
    9:  (30, 147, 255),   # blue
    10: (136, 216, 241),  # cyan
    11: (255, 220, 0),    # yellow
    12: (255, 133, 27),   # orange
    13: (146, 18, 49),    # maroon
    14: (79, 204, 48),    # green
    15: (163, 86, 214),   # purple
}
ARC_COLOR_NAMES = {
    0: "white",
    1: "light_gray",
    2: "gray",
    3: "dark_gray",
    4: "very_dark_gray",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "cyan",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}

def frame_to_data_url(
    frame,
    save_path=None):
    frame = np.asarray(frame)

    h, w = frame.shape

    rgb = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for color, rgb_value in ARC_PALETTE.items():
        rgb[frame == color] = rgb_value

    image = Image.fromarray(rgb)

    # ARC grids are tiny.
    # Upscale without smoothing.
    # image = image.resize(
    #     (w * 8, h * 8),
    #     Image.Resampling.NEAREST,
    # )

    buffer = io.BytesIO()

    # Optional debug image
    if save_path is not None:
        image.save(save_path)

    image.save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )
# detect and extract objects to GameObject

def object_hash(
    pixels: set[tuple[int, int]],
    color: int,) -> str:

    min_y = min(y for y, x in pixels)
    min_x = min(x for y, x in pixels)

    normalized = sorted(
        (y - min_y, x - min_x)
        for y, x in pixels
    )

    payload = repr(
        (color, normalized)
    ).encode()

    return hashlib.sha1(
        payload
    ).hexdigest()[:16]

def group_objects_by_hash(objects: list[GameObject],):
    groups = defaultdict(list)

    for obj in objects:
        groups[obj.shape_hash].append(
            obj.id
        )

    return {
        shape_hash: ids
        for shape_hash, ids
        in groups.items()
        if len(ids) >= 2
    }

def extract_objects(frame) -> list[GameObject]:
    frame = np.asarray(frame)

    # You may need to change this depending on ARC-AGI-3 frames.
    # background = 4

    height, width = frame.shape
    visited = set()
    objects = []

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    object_id = 0

    for y in range(height):
        for x in range(width):

            if (y, x) in visited:
                continue

            color = int(frame[y, x])

            # if color == background:
            #     continue

            queue = deque([(y, x)])
            visited.add((y, x))

            pixels = set()

            while queue:
                cy, cx = queue.popleft()
                pixels.add((cy, cx))

                for dy, dx in directions:
                    ny = cy + dy
                    nx = cx + dx

                    if not (0 <= ny < height and 0 <= nx < width):
                        continue

                    if (ny, nx) in visited:
                        continue

                    if frame[ny, nx] != color:
                        continue

                    visited.add((ny, nx))
                    queue.append((ny, nx))

            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]

            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)

            obj = GameObject(id=object_id,color=color,pixels=pixels,
                bbox=(min_y,min_x,max_y,max_x,),
                center=(sum(ys)/len(ys),sum(xs)/len(xs),),
                shape_hash=object_hash(pixels,color,),
                # Temporary until we implement
                # Duck-style contour/enclosure.
                boundary=[],
                children=[],
            )

            objects.append(obj)
            object_id += 1

    return objects

def group_movements(movements):

    groups = defaultdict(list)

    for movement in movements:
        groups[movement.delta].append(movement)

    return groups

def maximum_assignment(scores):
    """Global one-to-one assignment; dummy columns leave tracks unmatched."""
    if not scores:
        return []
    n, real_columns = len(scores), len(scores[0])
    costs = [[-score for score in row] + [0.0] * n for row in scores]
    m = real_columns + n
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum, used = [float('inf')] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float('inf'), 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, real_columns + 1) if p[j]]


class ObjectTracker:
    """Conservative component tracking; appearance is evidence, not identity."""
    MIN_SCORE = 0.60
    AMBIGUITY_MARGIN = 0.08
    MAX_MISSING = 2

    def __init__(self, episode_id=0):
        self.episode_id = episode_id
        self.next_id = 0
        self.tracks = {}
        self.missing = {}
        self.retired = set()
        self.events = []

    @staticmethod
    def score(old, new, delta=None, missing=False):
        predicted = translate_pixels(old.pixels, delta or (0, 0))
        # A blocked action is also a valid explanation for an unchanged object.
        overlap = max(
            len(predicted & new.pixels) / len(predicted | new.pixels),
            len(old.pixels & new.pixels) / len(old.pixels | new.pixels),
        )
        dx, dy = delta or (0, 0)
        distance = min(
            abs(old.center[0] + dy - new.center[0]) + abs(old.center[1] + dx - new.center[1]),
            abs(old.center[0] - new.center[0]) + abs(old.center[1] - new.center[1]),
        )
        if distance > 15 or (missing and (old.shape_hash != new.shape_hash or distance > 5)):
            return -1.0
        a, b = normalized_shape(old), normalized_shape(new)
        shape = len(a & b) / len(a | b)
        size = min(len(a), len(b)) / max(len(a), len(b))
        appearance = 0.20 * shape + 0.10 * size + 0.10 * (old.color == new.color)
        return max(0.70 * overlap + 0.30 * size,
                   appearance + 0.45 * max(0, 1 - distance / 20))

    def update(self, objects, controlled_tracks=(), delta=None):
        self.events = []
        previous = [obj for key, obj in self.tracks.items()
                    if key not in self.retired and self.missing[key] <= self.MAX_MISSING]
        # Detect geometric split/merge groups before one-to-one assignment.
        overlap_edges = {}
        for i, old in enumerate(previous):
            if self.missing[old.track_id]:
                continue
            poses = [old.pixels]
            if old.track_id in controlled_tracks and delta is not None:
                poses.append(translate_pixels(old.pixels, delta))
            # A blocked move may still trigger a color/shape transformation.
            pixels = max(poses, key=lambda pose: max(
                (len(pose & new.pixels) / len(pose | new.pixels) for new in objects),
                default=0.0,
            ))
            for j, new in enumerate(objects):
                overlap = len(pixels & new.pixels)
                if overlap / min(len(pixels), len(new.pixels)) >= 0.8:
                    overlap_edges[i, j] = overlap
        lineage_old, lineage_new = set(), {}
        for i, old in enumerate(previous):
            children = [j for oi, j in overlap_edges if oi == i
                        and len(objects[j].pixels) <= 0.8 * len(old.pixels)]
            if len(children) > 1:
                area = sum(len(objects[j].pixels) for j in children)
                coverage = sum(overlap_edges[i, j] for j in children)
                if 0.8 <= area / len(old.pixels) <= 1.25 and coverage / len(old.pixels) >= 0.8:
                    lineage_old.add(i)
                    for j in children:
                        lineage_new.setdefault(j, set()).add(old.track_id)
        for j, new in enumerate(objects):
            parents = [i for i, nj in overlap_edges if nj == j
                       and len(previous[i].pixels) <= 0.8 * len(new.pixels)]
            if len(parents) > 1:
                area = sum(len(previous[i].pixels) for i in parents)
                coverage = sum(overlap_edges[i, j] for i in parents)
                if 0.8 <= area / len(new.pixels) <= 1.25 and coverage / len(new.pixels) >= 0.8:
                    lineage_old.update(parents)
                    lineage_new.setdefault(j, set()).update(previous[i].track_id for i in parents)
        scores = [[
            self.score(old, new, delta if old.track_id in controlled_tracks else None,
                       bool(self.missing[old.track_id]))
            if i not in lineage_old and j not in lineage_new else -1.0
            for j, new in enumerate(objects)] for i, old in enumerate(previous)]
        matched_old, matched_new, uncertain = set(), set(), set()
        for i, j in maximum_assignment([
            [score if score >= self.MIN_SCORE else -1.0 for score in row] for row in scores
        ]):
            score = scores[i][j]
            if score < self.MIN_SCORE:
                continue
            competitors = [s for k, s in enumerate(scores[i]) if k != j]
            competitors += [row[j] for k, row in enumerate(scores) if k != i]
            if competitors and score - max(competitors) < self.AMBIGUITY_MARGIN:
                uncertain.add(j)
                continue
            objects[j].track_id = previous[i].track_id
            objects[j].tracking_confidence = score
            objects[j].tracking_status = 'matched'
            objects[j].parent_track_ids = previous[i].parent_track_ids
            matched_old.add(i)
            matched_new.add(j)
        for i, old in enumerate(previous):
            if i in lineage_old:
                self.retired.add(old.track_id)
            elif i not in matched_old:
                self.missing[old.track_id] += 1
                # Continued ambiguous observations are not an absence. Keep
                # competing identities alive until evidence separates them.
                if any(score >= self.MIN_SCORE for score in scores[i]):
                    self.missing[old.track_id] = min(self.missing[old.track_id], self.MAX_MISSING)
                self.events.append({'type': 'unresolved', 'track_id': old.track_id})
        for j, obj in enumerate(objects):
            if j not in matched_new:
                # Do not mint a confident replacement for an ambiguous candidate.
                ambiguous = j in uncertain or any(row[j] >= self.MIN_SCORE for row in scores)
                if ambiguous and j not in lineage_new:
                    obj.tracking_status = 'ambiguous'
                    obj.tracking_confidence = 0.0
                    continue
                obj.track_id = f'{self.episode_id}:{self.next_id}'
                self.next_id += 1
                obj.tracking_status = 'lineage' if j in lineage_new else 'new'
                obj.tracking_confidence = 1.0
                obj.parent_track_ids = tuple(sorted(lineage_new.get(j, ())))
                if obj.parent_track_ids:
                    self.events.append({'type': 'split_or_merge', 'track_id': obj.track_id,
                                        'parent_track_ids': obj.parent_track_ids})
            self.tracks[obj.track_id] = obj
            self.missing[obj.track_id] = 0
        return objects


def match_objects(before, after):
    by_track = {obj.track_id: obj for obj in after if obj.track_id is not None}
    matches = [(old, by_track.get(old.track_id) if old.track_id is not None else None)
               for old in before]
    matched = {new.id for _, new in matches if new is not None}
    return matches, [obj for obj in after if obj.id not in matched]

def describe_transition(
    before,
    action,
    after,
    ) -> Transition:

    matches, appeared = match_objects(
        before,
        after,
    )

    moved = []
    disappeared = []
    changed = []

    for old, new in matches:

        if new is None:
            disappeared.append(old)
            continue
        old_shape = normalized_shape(old)
        new_shape = normalized_shape(new)
        dy = new.center[0] - old.center[0]
        dx = new.center[1] - old.center[1]
            
        # Position and appearance changes are independent observations.
        rounded_dx = round(dx)
        rounded_dy = round(dy)

        if (
            abs(dx - rounded_dx) < 1e-6
            and abs(dy - rounded_dy) < 1e-6
            and (rounded_dx != 0 or rounded_dy != 0)
        ):
            moved.append(
                Movement(
                    object_id=old.id,
                    before=old,
                    after=new,
                    delta=(rounded_dx, rounded_dy),
                )
            )
        if old_shape != new_shape or old.color != new.color:
            changed.append((old, new))

    return Transition(
        action=action,
        moved_objects=moved,
        appeared_objects=appeared,
        disappeared_objects=disappeared,
        changed_objects=changed,
        global_changes={},
    )

def normalized_shape(obj):
    min_y = min(y for y, x in obj.pixels)
    min_x = min(x for y, x in obj.pixels)

    return frozenset(
        (y - min_y, x - min_x)
        for y, x in obj.pixels
    )

def normalize_object_ids(value):
    # Missing object_ids
    if value is None:
        return []

    # Correct format already
    if isinstance(value, list):
        return [
            int(x)
            for x in value
            if isinstance(x, (int, float))
        ]

    # Single ID accidentally returned
    if isinstance(value, (int, float)):
        return [int(value)]

    # Stringified list:
    # "[7, 8, 9]"
    if isinstance(value, str):

        text = value.strip()

        if not text:
            return []

        # First try proper JSON decoding.
        try:
            decoded = json.loads(text)

            if decoded != value:
                return normalize_object_ids(
                    decoded
                )

        except json.JSONDecodeError:
            pass

        # Fallback for things like:
        # "7, 8, 9"
        # "[7 8 9]"
        import re

        matches = re.findall(
            r"-?\d+",
            text,
        )

        if matches:
            normalized = [
                int(x)
                for x in matches
            ]

            print(
                "[TOOL NORMALIZE] "
                f"object_ids {value!r} "
                f"-> {normalized}",
                flush=True,
            )

            return normalized

        raise ValueError(
            f"Could not parse object_ids: {value!r}"
        )

    raise TypeError(
        "object_ids must be list, integer, "
        f"or string; got {type(value).__name__}"
    )

def bbox_contains(outer, inner):
    oy1, ox1, oy2, ox2 = outer.bbox
    iy1, ix1, iy2, ix2 = inner.bbox

    return (
        oy1 <= iy1
        and ox1 <= ix1
        and oy2 >= iy2
        and ox2 >= ix2
    )
def bbox_area(obj):
    y1, x1, y2, x2 = obj.bbox

    return (
        (y2 - y1 + 1)
        * (x2 - x1 + 1)
    )

def generate_containment_relationships(
    objects: list[GameObject],) -> list[SpatialRelation]:

    relationships = []

    for inner in objects:
        containers = []
        for outer in objects:
            if inner.id == outer.id:
                continue
            if bbox_contains(outer, inner):
                containers.append(outer)
        if not containers:
            continue

        # Choose the smallest enclosing bbox.
        nearest_container = min(
            containers,
            key=bbox_area,
        )

        relationships.append(
            SpatialRelation(
                relation="BBOX_INSIDE",
                a=inner.id,
                b=nearest_container.id,
            )
        )
    return relationships

def analyze_scene(
    frame,
    objects,
    controlled_entity,
    action_vectors,) -> SceneModel:
    return None

EXPECTED_GOAL_TYPES = {
    "reach_object",
    "match_pattern",
    "collect_objects",
    "activate_object",
    "move_into_region",
    "transform_shape",
    "unknown",
}


def normalize_scene_analysis(analysis):
    if not isinstance(analysis, dict):
        raise TypeError(
            f"scene analysis must be dict, got {type(analysis).__name__}"
        )

    # -------------------------------------------------
    # Repair malformed form produced by some local VLMs:
    #
    # {
    #   "goal_scores":
    #       "{\"reach_object\": {...},
    #          \"important_objects\": [...],
    #          \"wall_candidates\": [...]}"
    # }
    #
    # In that case the string is actually the COMPLETE
    # scene object, not just goal_scores.
    # -------------------------------------------------
    if (
        set(analysis.keys()) == {"goal_scores"}
        and isinstance(analysis["goal_scores"], str)
    ):
        raw = analysis["goal_scores"].strip()

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = None

        if isinstance(decoded, dict):

            # The VLM accidentally placed the whole
            # scene analysis inside goal_scores.
            if (
                "goal_scores" in decoded
                and "important_objects" in decoded
                and "wall_candidates" in decoded
            ):
                print(
                    "[SCENE NORMALIZE] "
                    "unwrapped complete scene object "
                    "from goal_scores string",
                    flush=True,
                )

                analysis = decoded

            # Another malformed form seen in your log:
            #
            # {
            #   "goal_scores": {
            #       "reach_object": ...,
            #       ...
            #       "important_objects": [...],
            #       "wall_candidates": [...]
            #   }
            # }
            #
            # Here those top-level fields were placed
            # inside the goal_scores object.
            elif (
                "important_objects" in decoded
                and "wall_candidates" in decoded
            ):
                print(
                    "[SCENE NORMALIZE] "
                    "promoting misplaced scene fields",
                    flush=True,
                )

                important_objects = decoded.pop(
                    "important_objects"
                )

                wall_candidates = decoded.pop(
                    "wall_candidates"
                )

                analysis = {
                    "goal_scores": decoded,
                    "important_objects": important_objects,
                    "wall_candidates": wall_candidates,
                }

    required_top_level = {
        "wall_candidates",
        "ui_candidates",
        "important_objects",
        "goal_scores",
    }

    missing = required_top_level - set(analysis)

    if missing:
        raise ValueError(
            f"scene analysis missing fields: {sorted(missing)}"
        )

    if not isinstance(analysis["wall_candidates"], list):
        raise TypeError(
            "wall_candidates must be a list"
        )
    if not isinstance(analysis["ui_candidates"],list,):
        raise TypeError(
            "ui_candidates must be a list"
        )
    if not isinstance(analysis["important_objects"], list):
        raise TypeError(
            "important_objects must be a list"
        )

    goal_scores = analysis["goal_scores"]

    # Some local models return the nested object as
    # a JSON-encoded string.
    if isinstance(goal_scores, str):
        try:
            goal_scores = json.loads(goal_scores)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "goal_scores was returned as an invalid JSON string"
            ) from exc

    # Some local models incorrectly wrap the object
    # in a one-element array:
    #
    # "goal_scores": [{...}]
    #
    # Unwrap this safely.
    if isinstance(goal_scores, list):
        if (
            len(goal_scores) == 1
            and isinstance(goal_scores[0], dict)
        ):
            print(
                "[SCENE NORMALIZE] unwrapping "
                "goal_scores singleton list",
                flush=True,
            )
            goal_scores = goal_scores[0]
        else:
            raise TypeError(
                "goal_scores must be an object; "
                f"received list with {len(goal_scores)} elements"
            )

    if not isinstance(goal_scores, dict):
        raise TypeError(
            "goal_scores must be dict, "
            f"got {type(goal_scores).__name__}"
        )

    missing_goals = (
        EXPECTED_GOAL_TYPES
        - set(goal_scores)
    )

    if missing_goals:
        raise ValueError(
            "goal_scores missing goal types: "
            f"{sorted(missing_goals)}"
        )

    for goal_type in EXPECTED_GOAL_TYPES:

        result = goal_scores[goal_type]

        if not isinstance(result, dict):
            raise TypeError(
                f"goal_scores[{goal_type!r}] "
                "must be an object"
            )

        required_fields = {
            "confidence",
            "target_ids",
            "evidence",
        }

        missing_fields = (
            required_fields
            - set(result)
        )

        if missing_fields:
            raise ValueError(
                f"{goal_type} missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not isinstance(
            result["target_ids"],
            list,
        ):
            raise TypeError(
                f"{goal_type}.target_ids "
                "must be a list"
            )

        if not isinstance(
            result["confidence"],
            (int, float),
        ):
            raise TypeError(
                f"{goal_type}.confidence "
                "must be numeric"
            )

        if not isinstance(
            result["evidence"],
            str,
        ):
            raise TypeError(
                f"{goal_type}.evidence "
                "must be a string"
            )

    # Work on a shallow copy rather than unexpectedly
    # mutating the raw model result.
    analysis = dict(analysis)
    analysis["goal_scores"] = goal_scores

    return analysis

def validate_scene_analysis(
    analysis,
    controlled_ids,
    objects,
    traversable_color_evidence,):

    analysis = normalize_scene_analysis(analysis)

    controlled_ids = set(controlled_ids)

    strongly_traversable_colors = {
        color
        for color, count
        in traversable_color_evidence.items()
        if count >= 2
    }

    objects_by_id = {
        obj.id: obj
        for obj in objects
    }

    ui_ids = set(
        analysis.get(
            "ui_candidates",
            [],
        )
    )

    # Controlled entity cannot be a wall.
    analysis["wall_candidates"] = [
        obj_id
        for obj_id in analysis["wall_candidates"]
        if obj_id not in controlled_ids
    ]

    # An object whose color has repeatedly been
    # traversed should not be considered a wall.
    analysis["wall_candidates"] = [
        obj_id
        for obj_id
        in analysis["wall_candidates"]
        if (
            obj_id not in objects_by_id
            or objects_by_id[obj_id].color
            not in strongly_traversable_colors
        )
    ]

    # Exclude controlled components and UI from important objects.
    analysis["important_objects"] = [
        obj_id
        for obj_id in analysis["important_objects"]
        if obj_id not in controlled_ids
           and obj_id not in ui_ids
    ]

    # Remove controlled components from
    # every goal's target list.
    for goal_type, result in (
        analysis["goal_scores"].items()
    ):

        original_targets = set(
            result["target_ids"]
        )

        cleaned_targets = (
            original_targets
            - controlled_ids
            - ui_ids
        )
        cleaned_targets = {
            obj_id for obj_id in cleaned_targets
            if obj_id in objects_by_id and objects_by_id[obj_id].track_id is not None
        }

        result["target_ids"] = sorted(
            cleaned_targets
        )

        if (
            original_targets
            and not cleaned_targets
        ):
            result["confidence"] = 0.0

            result["evidence"] += (
                " Targets were removed because they were controlled, UI, "
                "missing, or ambiguously tracked components."
            )

    return analysis

def build_composite_regions(
    relationships,):
    contents = defaultdict(list)

    for relation in relationships:
        contents[relation.b].append(
            relation.a
        )

    return [
        CompositeRegion(
            container_id=container,
            content_ids=children,
        )
        for container, children
        in contents.items()
    ]

def get_controlled_pixels(
    objects,
    controlled_ids,):
    pixels = set()

    for obj in objects:
        if obj.id in controlled_ids:
            pixels |= obj.pixels

    return pixels

def translate_pixels(
    pixels,
    delta,):
    dx, dy = delta

    return {
        (y + dy, x + dx)
        for y, x in pixels
    }

def detect_ui_candidates(
    frame,
    objects,
    controlled_ids,):
    height, width = frame.shape

    result = []

    for obj in objects:

        if obj.id in controlled_ids:
            continue

        y1, x1, y2, x2 = obj.bbox

        touches_edge = (
            y1 == 0
            or x1 == 0
            or y2 == height - 1
            or x2 == width - 1
        )

        near_edge = (
            y1 <= 2
            or x1 <= 2
            or y2 >= height - 3
            or x2 >= width - 3
        )

        if touches_edge or near_edge:
            result.append(obj.id)

    return result

def destination_colors(
    frame,
    controlled_pixels,
    delta,):
    future_pixels = translate_pixels(
        controlled_pixels,
        delta,
    )

    # Only cells newly entered by the entity.
    entering_pixels = (
        future_pixels
        - controlled_pixels
    )

    colors = set()

    height, width = frame.shape

    for y, x in entering_pixels:

        if not (
            0 <= y < height
            and 0 <= x < width
        ):
            continue

        colors.add(
            int(frame[y, x])
        )

    return colors

def select_primary_goal(
    scene_analysis,
    min_confidence=0.25,
    goal_penalties=None,):
    scores = scene_analysis["goal_scores"]

    unknown_confidence = (
        scores["unknown"]["confidence"]
    )

    candidates = []

    for goal_type, result in scores.items():

        if goal_type == "unknown":
            continue

        if not result["target_ids"]:
            continue

        penalty = (goal_penalties or {}).get(goal_type, 0.0)
        # None means this experiment is on hold until new evidence appears.
        if penalty is None:
            continue

        candidates.append(
            (
                max(0.0, result["confidence"] - penalty),
                goal_type,
                result["target_ids"],
            )
        )

    if not candidates:
        return None

    confidence, goal_type, targets = max(
        candidates,
        key=lambda x: x[0],
    )

    # Don't commit if the VLM itself is too uncertain.
    if confidence < min_confidence:
        return None

    if confidence <= unknown_confidence:
        return None

    return {
        "type": goal_type,
        "confidence": confidence,
        "target_ids": targets,
    }

def get_object_pixels(
    objects,
    object_ids,):
    ids = set(object_ids)
    pixels = set()

    for obj in objects:
        if obj.id in ids:
            pixels |= obj.pixels

    return pixels

def pixel_distance(
    pixels_a,
    pixels_b,):
    if not pixels_a or not pixels_b:
        return float("inf")

    return min(
        abs(y1 - y2) + abs(x1 - x2)
        for y1, x1 in pixels_a
        for y2, x2 in pixels_b
    )

def choose_goal_action(
    frame,
    objects,
    controlled_ids,
    target_ids,
    action_vectors,
    scene_analysis,
    traversable_color_evidence,):
    controlled_pixels = get_object_pixels(
        objects,
        controlled_ids,
    )

    target_pixels = get_object_pixels(
        objects,
        target_ids,
    )

    if not controlled_pixels or not target_pixels:
        return None

    current_distance = pixel_distance(
        controlled_pixels,
        target_pixels,
    )

    # Colors with repeated experimental evidence.
    traversable_colors = {
        color
        for color, count
        in traversable_color_evidence.items()
        if count >= 2
    }

    # VLM wall objects are soft evidence,
    # but useful for planning.
    wall_pixels = get_object_pixels(
        objects,
        scene_analysis["wall_candidates"],
    )

    height, width = frame.shape

    candidates = []

    for action, delta in action_vectors.items():

        future_pixels = translate_pixels(
            controlled_pixels,
            delta,
        )

        # Reject out-of-bounds movement.
        if any(
            not (0 <= y < height and 0 <= x < width)
            for y, x in future_pixels
        ):
            continue

        entering_pixels = (
            future_pixels
            - controlled_pixels
        )

        # Don't move into a known wall unless that
        # wall is deliberately the target.
        wall_collision = (
            entering_pixels
            & wall_pixels
            - target_pixels
        )

        if wall_collision:
            continue

        unknown_cells = 0

        for y, x in entering_pixels:

            # Entering the goal itself is allowed.
            if (y, x) in target_pixels:
                continue

            color = int(frame[y, x])

            if color not in traversable_colors:
                unknown_cells += 1

        future_distance = pixel_distance(
            future_pixels,
            target_pixels,
        )

        progress = (
            current_distance
            - future_distance
        )

        # Strong preference for approaching the goal,
        # mild penalty for entering unverified terrain.
        score = (
            progress * 10
            - unknown_cells * 2
        )

        candidates.append(
            (
                score,
                action,
                progress,
                future_distance,
                unknown_cells,
            )
        )

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda x: x[0],
    )

    # logging candidates
    for candidate in candidates:
        (
            candidate_score,
            candidate_action,
            candidate_progress,
            candidate_distance,
            candidate_unknown,
        ) = candidate

        print(
            f"[GOAL CANDIDATE] "
            f"{candidate_action.name}: "
            f"score={candidate_score}, "
            f"progress={candidate_progress}, "
            f"distance={candidate_distance}, "
            f"unknown={candidate_unknown}",
            flush=True,
        )
        
    score, action, progress, distance, unknown = best

    print(
        f"PLANNER: {action.name}, "
        f"score={score}, "
        f"progress={progress}, "
        f"distance={distance}, "
        f"unknown_cells={unknown}"
    )

    return action

def choose_exploration_action(
    frame,
    objects,
    controlled_ids,
    action_vectors,
    scene_analysis,
    traversable_color_evidence,
    state_visit_counts=None,):

    if state_visit_counts is None:
        state_visit_counts = {}

    controlled_pixels = get_object_pixels(
        objects,
        controlled_ids,
    )

    if not controlled_pixels:
        return None

    traversable_colors = {
        color
        for color, count
        in traversable_color_evidence.items()
        if count >= 2
    }

    wall_pixels = get_object_pixels(
        objects,
        scene_analysis["wall_candidates"],
    )

    height, width = frame.shape

    candidates = []

    for action, delta in action_vectors.items():

        future_pixels = translate_pixels(
            controlled_pixels,
            delta,
        )

        # Outside board = invalid.
        if any(
            not (0 <= y < height and 0 <= x < width)
            for y, x in future_pixels
        ):
            continue

        entering_pixels = (
            future_pixels
            - controlled_pixels
        )

        # Avoid known walls.
        if entering_pixels & wall_pixels:
            continue

        known_traversable = 0
        unknown = 0

        for y, x in entering_pixels:
            color = int(frame[y, x])

            if color in traversable_colors:
                known_traversable += 1
            else:
                unknown += 1

        future_state = frozenset(future_pixels)

        visit_count = state_visit_counts.get(future_state,0,)

        score = (
            known_traversable
            + unknown * 0.25
            - visit_count * 5.0
        )

        print(
            "[EXPLORE CANDIDATE]",
            action.name,
            f"score={score}",
            f"visited={visit_count}",
            flush=True,
        )

        candidates.append(
            (score, action)
        )

    if not candidates:
        return None

    _, action = max(
        candidates,
        key=lambda x: x[0],
    )

    print(
        f"EXPLORATION: {action.name}"
    )

    return action

def choose_goal_action_bfs(
    frame,
    objects,
    controlled_ids,
    target_ids,
    action_vectors,
    traversable_color_evidence,
    failed_moves,
    max_depth=30,):
    controlled_pixels = get_controlled_pixels(
        objects,
        controlled_ids,
    )

    target_pixels = get_object_pixels(
        objects,
        target_ids,
    )

    if (
        not controlled_pixels
        or not target_pixels
    ):
        return None

    traversable_colors = {
        color
        for color, count
        in traversable_color_evidence.items()
        if count >= 2
    }

    start = frozenset(
        controlled_pixels
    )

    height, width = frame.shape

    def legal(pixels):
        for y, x in pixels:

            if not (
                0 <= y < height
                and 0 <= x < width
            ):
                return False

            # Goal may be entered.
            if (y, x) in target_pixels:
                continue

            # Current player's cells are
            # logically vacated floor.
            if (y, x) in controlled_pixels:
                continue

            if (
                int(frame[y, x])
                not in traversable_colors
            ):
                return False

        return True

    queue = deque([(start,None,0,)])

    visited = {start}

    while queue:

        pixels, first_action, depth = (queue.popleft())

        if pixels & target_pixels:
            print(
                "[BFS] path found, "
                f"first={first_action}",
                flush=True,
            )

            return first_action

        if depth >= max_depth:
            continue

        for action, delta in (
            action_vectors.items()
        ):
            failed_key = (
                pixels,
                action,
            )

            if failed_key in failed_moves:
                print("[BFS] skipping known failed move:",action.name,flush=True,)
                continue
            
            next_pixels = frozenset(translate_pixels(pixels,delta,))

            if next_pixels in visited:
                continue

            if not legal(next_pixels):
                continue

            visited.add(next_pixels)

            queue.append(
                (
                    next_pixels,
                    (first_action if first_action is not None else action),
                    depth + 1,
                )
            )

    print(
        "[BFS] no verified path",
        flush=True,
    )

    return None

def capture_goal_target_tracks(
    scene_analysis,
    objects,):
    if not isinstance(scene_analysis, dict):
        print(
            "[TARGET TRACK WARNING] "
            "scene_analysis is not a dict:",
            type(scene_analysis).__name__,
            flush=True,
        )
        return {}

    goal_scores = scene_analysis.get(
        "goal_scores",
        {}
    )

    if not isinstance(goal_scores, dict):
        print(
            "[TARGET TRACK WARNING] "
            "goal_scores is not a dict:",
            type(goal_scores).__name__,
            flush=True,
        )
        return {}

    objects_by_id = {
        obj.id: obj
        for obj in objects
    }

    result = {}

    for goal_type, goal in (
        scene_analysis["goal_scores"].items()
    ):
        result[goal_type] = [
            objects_by_id[obj_id].track_id
            for obj_id in goal["target_ids"]
            if obj_id in objects_by_id and objects_by_id[obj_id].track_id is not None
        ]

    return result

def resolve_goal_target_ids(
    objects,
    target_tracks,):
    tracks = set(target_tracks)
    resolved = {obj.track_id: obj.id for obj in objects if obj.track_id in tracks}
    # Never silently plan against only a subset of an unresolved target group.
    return [resolved[track] for track in sorted(tracks)] if tracks <= resolved.keys() else []


def bbox_gap(a, b):
    """Chebyshev gap between two inclusive bounding boxes."""
    ay1, ax1, ay2, ax2 = a.bbox
    by1, bx1, by2, bx2 = b.bbox
    return max(
        0,
        ay1 - by2 - 1,
        by1 - ay2 - 1,
        ax1 - bx2 - 1,
        bx1 - ax2 - 1,
    )


def expand_goal_target_region(
    objects,
    target_ids,
    controlled_ids,
    ui_ids,
    wall_ids,
    traversable_color_evidence,
    max_gap=1,
    max_component_pixels=16,
    max_region_pixels=32,
):
    """Include small, adjoining marker components required for target contact.

    The selected target remains the persistent identity anchor. Expansion only
    affects collision legality for this frame, so nearby UI, known walls, the
    player, and confirmed floor cannot be silently reclassified as a target.
    """
    by_id = {obj.id: obj for obj in objects}
    region_ids = {obj_id for obj_id in target_ids if obj_id in by_id}
    if not region_ids:
        return []

    traversable_colors = {
        color for color, count in traversable_color_evidence.items()
        if count >= 2
    }
    protected_ids = set(controlled_ids) | set(ui_ids) | set(wall_ids)

    while True:
        region_pixels = sum(len(by_id[obj_id].pixels) for obj_id in region_ids)
        candidates = []
        for candidate in objects:
            if candidate.id in region_ids or candidate.id in protected_ids:
                continue
            if candidate.track_id is None or candidate.color in traversable_colors:
                continue
            if len(candidate.pixels) > max_component_pixels:
                continue
            if region_pixels + len(candidate.pixels) > max_region_pixels:
                continue
            if any(
                bbox_gap(candidate, by_id[region_id]) <= max_gap
                for region_id in region_ids
            ):
                candidates.append(candidate.id)

        if not candidates:
            break
        region_ids.update(candidates)

    return sorted(region_ids)

def controlled_anchor(
    objects,
    controlled_ids,):
    pixels = get_controlled_pixels(
        objects,
        controlled_ids,
    )

    if not pixels:
        return None

    return (
        min(y for y, x in pixels),
        min(x for y, x in pixels),
    )

ARC_COLOR_CHARS = ("WwgGcBMPRbSYOrNp")
def frame_crop_ascii(
    frame,
    min_row,
    min_col,
    max_row,
    max_col,):
    frame = np.asarray(frame)

    min_row = max(
        0,
        int(min_row),
    )
    min_col = max(
        0,
        int(min_col),
    )

    max_row = min(
        frame.shape[0] - 1,
        int(max_row),
    )
    max_col = min(
        frame.shape[1] - 1,
        int(max_col),
    )

    lines = []

    for row in range(
        min_row,
        max_row + 1,
    ):
        line = "".join(
            ARC_COLOR_CHARS[
                int(frame[row, col])
            ]
            for col in range(
                min_col,
                max_col + 1,
            )
        )

        lines.append(line)

    return {
        "rows": [
            min_row,
            max_row,
        ],
        "cols": [
            min_col,
            max_col,
        ],
        "legend": {
            ARC_COLOR_CHARS[i]:
                ARC_COLOR_NAMES[i]
            for i in range(
                len(ARC_COLOR_CHARS)
            )
        },
        "ascii": "\n".join(lines),
    }

def inspect_scene(
    args,
    frame,
    objects,
    controlled_components,
    action_vectors,
    traversable_color_evidence,):
    # Do not mutate the original tool arguments.
    args = dict(args)

    if "object_ids" in args:
        args["object_ids"] = (
            normalize_object_ids(
                args["object_ids"]
            )
        )

    query = args["query"]

    if query == "summary":
        return {
            "shape": list(frame.shape),
            "component_count": len(objects),
            "controlled_ids": sorted(
                controlled_components
            ),
            "action_vectors": {
                action.name: list(delta)
                for action, delta
                in action_vectors.items()
            },
            "traversable_colors": dict(
                traversable_color_evidence
            ),
        }

    if query == "controlled":
        return [
            {
                "id": obj.id,
                "color": ARC_COLOR_NAMES[
                    obj.color
                ],
                "pixels": len(obj.pixels),
                "bbox": list(obj.bbox),
                "shape_hash": obj.shape_hash,
                "track_id": obj.track_id,
                "tracking_status": obj.tracking_status,
                "tracking_confidence": obj.tracking_confidence,
                "parent_track_ids": list(obj.parent_track_ids),
            }
            for obj in objects
            if obj.id in controlled_components
        ]
    if query == "component":
        ids = args.get("object_ids")

        if ids:
            ids = set(ids)

            selected_objects = [
                obj
                for obj in objects
                if obj.id in ids
            ]

        else:
            selected_objects = objects

        return [
            {
                "id": obj.id,
                "color": (
                    ARC_COLOR_NAMES[
                        obj.color
                    ]
                ),
                "pixels": len(obj.pixels),
                "bbox": list(obj.bbox),
                "center": list(obj.center),
                "shape_hash": obj.shape_hash,
                "track_id": obj.track_id,
                "tracking_status": obj.tracking_status,
                "tracking_confidence": obj.tracking_confidence,
                "parent_track_ids": list(obj.parent_track_ids),
            }
            for obj in selected_objects
        ]
    if query == "repeated_shapes":
        return group_objects_by_hash(
            objects
        )

    if query == "crop":
        r1 = args["min_row"]
        c1 = args["min_col"]
        r2 = args["max_row"]
        c2 = args["max_col"]

        return frame_crop_ascii(
            frame,
            r1,
            c1,
            r2,
            c2,
        )

    return {
        "error": f"unknown query {query}"
    }

def get_frontier_actions(
    state,
    action_vectors,
    tested_actions,
    blocked_actions,):
    """
    Return actions from this state that have never
    been experimentally tested.
    """

    frontier_actions = []

    for action in action_vectors:

        if action in tested_actions.get(
            state,
            set(),
        ):
            continue

        if (
            state,
            action,
        ) in blocked_actions:
            continue

        frontier_actions.append(action)

    return frontier_actions

def plan_to_nearest_frontier(
    start_state,
    action_vectors,
    transition_graph,
    tested_actions,
    blocked_actions,
    previous_action=None,):
    """
    Search the learned transition graph for the
    closest reachable state that still has an
    untested action.

    Returns a list of actions.
    """

    if not start_state:
        return None

    queue = deque([
        (
            start_state,
            [],
        )
    ])

    visited = {
        start_state
    }

    while queue:

        state, path = queue.popleft()

        frontier_actions = (
            get_frontier_actions(
                state=state,
                action_vectors=action_vectors,
                tested_actions=tested_actions,
                blocked_actions=blocked_actions,
            )
        )

        # If this state has an unknown transition,
        # travel here and then test one.
        if frontier_actions:

            # If we travelled through the learned graph
            # to reach this frontier, the last action in
            # the path is the action we just used.
            #
            # If this is the starting state, use the real
            # previous action executed by the agent.
            last_action = (
                path[-1]
                if path
                else previous_action
            )

            reverse = inverse_action(
                last_action,
                action_vectors,
            )

            # Prefer testing a new direction rather than
            # immediately reversing the last movement.
            preferred = [
                action
                for action in frontier_actions
                if action != reverse
            ]

            if preferred:
                exploration_action = preferred[0]
            else:
                exploration_action = frontier_actions[0]

            return (
                path
                + [exploration_action]
            )

        # Otherwise move through transitions
        # we already know.
        for action, next_state in (
            transition_graph
            .get(state, {})
            .items()
        ):

            # Historical edges may use controls unavailable in this frame.
            if action not in action_vectors:
                continue

            if next_state in visited:
                continue

            visited.add(next_state)

            queue.append(
                (
                    next_state,
                    path + [action],
                )
            )

    return None

def inverse_action(
    action,
    action_vectors,):
    if action is None:
        return None

    delta = action_vectors.get(action)

    if delta is None:
        return None

    dx, dy = delta

    inverse_delta = (
        -dx,
        -dy,
    )

    for candidate, candidate_delta in (
        action_vectors.items()
    ):
        if candidate_delta == inverse_delta:
            return candidate

    return None

def choose_frontier_action(
    objects,
    controlled_ids,
    action_vectors,
    transition_graph,
    tested_actions,
    blocked_actions,
    previous_action=None):
    current_pixels = get_controlled_pixels(
        objects,
        controlled_ids,
    )

    if not current_pixels:
        return None

    current_state = frozenset(
        current_pixels
    )

    plan = plan_to_nearest_frontier(
        start_state=current_state,
        action_vectors=action_vectors,
        transition_graph=transition_graph,
        tested_actions=tested_actions,
        blocked_actions=blocked_actions,
        previous_action=previous_action,
    )

    if not plan:
        print(
            "[FRONTIER] no reachable frontier",
            flush=True,
        )
        return None

    print(
        "[FRONTIER PLAN]",
        " -> ".join(
            action.name
            for action in plan
        ),
        flush=True,
    )

    return plan[0]

# formatting response
GOAL_RESULT = {
    "type": "object",
    "properties": {
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "target_ids": {
            "type": "array",
            "items": {
                "type": "integer"
            },
        },
        "evidence": {
            "type": "string"
        },
    },
    "required": [
        "confidence",
        "target_ids",
        "evidence",
    ],
    "additionalProperties": False,
}
SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "wall_candidates": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "ui_candidates": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "important_objects": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "goal_scores": {
          "type": "object",
          "properties": {
              "reach_object": GOAL_RESULT,
              "match_pattern": GOAL_RESULT,
              "collect_objects": GOAL_RESULT,
              "activate_object": GOAL_RESULT,
              "move_into_region": GOAL_RESULT,
              "transform_shape": GOAL_RESULT,
              "unknown": GOAL_RESULT,
          },
          "required": [
              "reach_object",
              "match_pattern",
              "collect_objects",
              "activate_object",
              "move_into_region",
              "transform_shape",
              "unknown",
          ],
          "additionalProperties": False,
        },
        "object_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "integer"},
                    "role": {
                        "type": "string",
                        "enum": [
                            "key",
                            "door",
                            "switch",
                            "exit",
                            "collectible",
                            "hazard",
                            "portal",
                            "obstacle",
                            "marker",
                            "unknown"
                        ]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "evidence": {"type": "string"}
                },
                "required": [
                    "object_id",
                    "role",
                    "confidence",
                    "evidence"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "wall_candidates",
        "ui_candidates",
        "important_objects",
        "goal_scores",
    ],
    "additionalProperties": False,
}
INSPECT_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "enum": [
                "summary",
                "component",
                "repeated_shapes",
                "controlled",
                "crop",
            ],
        },
        # "object_id": {"type": "integer",},
        "object_ids": {"type": "array","items": {"type": "integer"},},
        "min_row": {"type": "integer",},
        "min_col": {"type": "integer",},
        "max_row": {"type": "integer",},
        "max_col": {"type": "integer",},
    },
    "required": ["query"],
    "additionalProperties": False,
}
SCENE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_scene",
            "description": (
                "Inspect deterministic structural information "
                "about the current ARC scene. Use this instead "
                "of guessing object relationships from the image."
            ),
            "parameters": INSPECT_SCENE_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_scene_analysis",
            "description": (
                "Submit the final scene interpretation when "
                "you have enough evidence."
            ),
            "parameters": SCENE_SCHEMA,
        },
    },
]

def analyze_scene_vlm(
    frame,
    objects,
    action_vectors,
    controlled_components,
    traversable_color_evidence,
    previous_feedback=None,
    reanalysis_reason=None,
    tracking_events=None,
    causal_observations=None,):
    image_url = frame_to_data_url(frame)
    controlled_description = [
        {
            "frame_id": obj.id,
            "color": ARC_COLOR_NAMES[
                obj.color
            ],
            "shape_hash": obj.shape_hash,
            "track_id": obj.track_id,
            "tracking_status": obj.tracking_status,
            "tracking_confidence": obj.tracking_confidence,
            "parent_track_ids": list(obj.parent_track_ids),
            "bbox": list(obj.bbox),
        }
        for obj in objects
        if obj.id
        in controlled_components
    ]
    verified_facts = {
        "tracking_events": tracking_events or [],
        "causal_observations": causal_observations or [],
        "frame_shape": list(np.asarray(frame).shape),
        "component_table": {
            "columns": ["frame_id", "color", "pixel_count", "bbox_yxyx", "shape_hash", "track_id", "tracking_status", "confidence", "parent_track_ids"],
            "rows": [
                [obj.id, ARC_COLOR_NAMES[obj.color], len(obj.pixels),
                 list(obj.bbox), obj.shape_hash, obj.track_id, obj.tracking_status,
                 obj.tracking_confidence, list(obj.parent_track_ids)]
                for obj in objects
            ],
        },
        "controlled_components": controlled_description,
        "action_vectors": {
            action.name: list(delta)
            for action, delta
            in action_vectors.items()
        },
        "traversable_colors":  [
            {
                "color_id": color,
                "color_name": (ARC_COLOR_NAMES[color]),
                "observations": count,
            }
            for color, count
            in traversable_color_evidence.items()
        ],
        "edge_ui_candidates":
        detect_ui_candidates(
            frame,
            objects,
            controlled_components,
        ),
    }

    initial_prompt = f"""
You are analyzing an unknown ARC-style interactive puzzle.

Verified experimental facts:

{json.dumps(
    verified_facts,
    indent=2,
)}

Component frame_id values are local to the current frame.
They may change after actions.

Do not use frame_id as persistent object identity.

track_id represents tracked continuity within this episode. shape_hash only
represents color and shape; identical objects can share it. Ambiguous components
have no track_id and must not be selected as targets. Missing tracks are unresolved,
not automatically collected. Split/merge descendants have fresh IDs with parent_track_ids.
Use current frame_id values when submitting tool arguments.

Connected components are low-level visual
components, not necessarily semantic objects.

Use inspect_scene selectively to answer
specific remaining questions.
The component table already provides all component IDs, colors, sizes,
bounding boxes (min_y, min_x, max_y, max_x), and shape hashes.
Do not request summary or component information already present above.

Prefer comparing several relevant components
in one inspection rather than inspecting every
component individually.

When you have a plausible world/goal model,
stop inspecting and call
submit_scene_analysis.

Some connected components may be user-interface elements rather
than gameplay objects.

UI elements often have one or more of these properties:

- Touch or closely follow the outer screen boundary.
- Stay in a fixed screen position while the controlled entity moves.
- Change size or appearance after actions without being interacted with.
- Look like health, energy, lives, score, inventory, or progress indicators.
- Are spatially separated from the main playable region.

Identify likely UI components in ui_candidates.

Do not include UI components as goal targets unless there is
strong evidence that the game explicitly requires interacting
with the interface itself.

The level may require multiple sequential interactions.

Do not assume the visible final-looking object is directly reachable.

Consider whether objects have semantic roles such as:
- key / collectible
- switch / activator
- locked door / barrier
- portal
- exit / destination
- hazard

A useful object may be a prerequisite rather than the final goal.

When an object disappears after the controlled entity contacts it,
treat this as evidence that it was collected or activated.

Look for resulting changes elsewhere in the gameplay area and infer
the next likely subgoal.

Causal observations report what changed immediately after the controlled
entity contacted a target region. An observation marked observed_once is an
association, not proof of a mechanic; repeated observations are stronger.
Treat UI changes as evidence about game state, not as movement targets unless
the game explicitly requires interacting with the UI.
You have a limited inspection budget.

Previous experimental feedback:

The records below aggregate all previous attempts by target and interaction.
A stalled experiment is not proof that the puzzle goal is false.
Inconclusive attempts did not establish success or failure. Entries marked
retry_requires_new_evidence are withheld by the action selector until new
navigation evidence or measurable progress appears. Changing the goal label
does not make the same movement experiment new.

{json.dumps(previous_feedback or [], indent=2)}

Reason analysis was requested again:

{reanalysis_reason or "initial analysis"}

If a previous goal/target hypothesis caused a loop,
unexpected reset, repeated states, or made no useful
progress, treat that as negative experimental evidence.

Do not simply repeat a failed hypothesis unless the
current frame provides new evidence supporting it.

Consider alternative goal types and alternative target
objects. Prefer hypotheses that can be tested with a
small number of actions.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": initial_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        }
    ]

    MAX_INSPECTION_STEPS = 6
    MAX_TOTAL_STEPS = 13
    MAX_NO_TOOL_RESPONSES = 3
    no_tool_responses = 0
    submission_only = False
    forced_tool_choice_supported = True
    for step in range(MAX_TOTAL_STEPS):
        remaining = (MAX_TOTAL_STEPS - step)
        if remaining == 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Only two tool steps remain. "
                        "If the available evidence is "
                        "sufficient, submit now. "
                        "Call submit_scene_analysis with your best complete analysis."
                    ),
                }
            )

        print(
            f"[VLM TOOL STEP {step + 1}]",
            flush=True,
        )
        if step >= MAX_INSPECTION_STEPS or no_tool_responses >= 2:
            submission_only = True

        if not submission_only:
            tools = SCENE_TOOLS
            tool_choice = "required" if no_tool_responses else "auto"
            enable_thinking = no_tool_responses == 0
        else:
            tools = [SCENE_TOOLS[1]]
            tool_choice = {
                "type": "function",
                "function": {"name": "submit_scene_analysis"},
            }
            enable_thinking = False
            messages.append({
                "role": "user",
                "content": (
                    "Inspection is finished. Call submit_scene_analysis now "
                    "with your best complete analysis. Do not answer in plain text."
                ),
            })
        if not forced_tool_choice_supported:
            tool_choice = "auto"
        try:
            response = client.chat.completions.create(
                model=LOCAL_MODEL,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0,
                max_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    },
                },
            )
        except BadRequestError as exc:
            # Some local servers only implement automatic tool choice.
            # Fall back once, within the same bounded request budget.
            detail = str(exc).lower()
            if (
                forced_tool_choice_supported
                and tool_choice != "auto"
                and ("tool_choice" in detail or "tool choice" in detail)
                and any(word in detail for word in ("unsupported", "not supported", "only", "must be"))
            ):
                forced_tool_choice_supported = False
                print("[VLM] forced tool choice unsupported; using bounded auto fallback", flush=True)
                continue
            raise

        message = (response.choices[0].message)

        reasoning = (
            getattr(message,"reasoning",None,)
            or getattr(message,"reasoning_content",None,)
            or ""
        )

        print("reasoning:",reasoning[:2000],flush=True,)

        tool_calls = (message.tool_calls or [])

        print("tool call count:",len(tool_calls),flush=True,)

        if not tool_calls:
            no_tool_responses += 1
            print(f"[VLM NO TOOL] {no_tool_responses}/{MAX_NO_TOOL_RESPONSES}", flush=True)
            if no_tool_responses >= MAX_NO_TOOL_RESPONSES:
                break
            # Do not append an empty assistant message or repeat its monologue.
            messages.append({
                "role": "user",
                "content": (
                    "Your last response made no tool call. Stop deliberating and "
                    "call inspect_scene for missing evidence or submit_scene_analysis now."
                ),
            })
            continue

        assistant_message = {"role": "assistant",}

        if message.content:
            assistant_message["content"] = (message.content)
        # if reasoning:
        #     assistant_message["reasoning"] = (reasoning)
        if tool_calls:
            assistant_message["tool_calls"] = [
                call.model_dump()
                for call in tool_calls
            ]

        messages.append(assistant_message)

        for call in tool_calls:
            name = call.function.name

            raw_args = call.function.arguments

            print(
                f"[RAW TOOL ARGS] "
                f"name={name} "
                f"id={call.id} "
                f"args={raw_args!r}",
                flush=True,
            )

            try:
                if isinstance(raw_args, dict):
                    args = raw_args
                else:
                    raw_args = str(raw_args or "").strip()

                    if not raw_args:
                        raise json.JSONDecodeError(
                            "empty tool arguments",
                            raw_args,
                            0,
                        )

                    args = json.loads(raw_args)

            except json.JSONDecodeError as exc:
                print(
                    "[TOOL ARG JSON ERROR]",
                    f"name={name}",
                    f"error={exc}",
                    f"raw={raw_args!r}",
                    flush=True,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {
                                "error": (
                                    "Your tool arguments were "
                                    "not valid JSON. "
                                    "Please call the tool again "
                                    "with valid JSON arguments."
                                )
                            }
                        ),
                    }
                )

                continue

            print(f"[TOOL] {name}: {args}",flush=True,)

            
            # -------------------------
            # Final submission
            # -------------------------
            if (name == "submit_scene_analysis"):
                submission_only = True
                try:
                    return (
                        validate_scene_analysis(
                            args,
                            controlled_components,
                            objects,
                            traversable_color_evidence,
                        )
                    )
                except (KeyError,TypeError,ValueError,) as exc:
                    print(
                        "[BAD SCENE SUBMISSION]",
                        repr(exc),
                        flush=True,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                call.id
                            ),
                            "content": json.dumps(
                                {
                                    "error": (
                                        "Scene analysis arguments were invalid: "
                                        f"{type(exc).__name__}: {exc}. "
                                        "Retry submit_scene_analysis with "
                                        "goal_scores, important_objects, and "
                                        "wall_candidates as top-level arguments."
                                    )
                                }
                            ),
                        }
                    )

                    continue
            # -------------------------
            # Inspection
            # -------------------------
            if name == "inspect_scene":
                try:
                    if submission_only:
                        raise ValueError("Inspection budget closed; call submit_scene_analysis")
                    result = inspect_scene(
                        args=args,
                        frame=frame,
                        objects=objects,
                        controlled_components=(controlled_components),
                        action_vectors=(action_vectors),
                        traversable_color_evidence=(traversable_color_evidence),
                    )
                except Exception as exc:
                    result = {
                        "error": (
                            f"inspect_scene failed: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    }
                print(
                    "[TOOL RESULT]",
                    json.dumps(result)[:3000],
                    flush=True,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            call.id
                        ),
                        "content": json.dumps(
                            result
                        ),
                    }
                )
                continue

            # Unknown tool
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        {
                            "error": (
                                f"Unknown tool: {name}"
                            )
                        }
                    ),
                }
            )
    print(
        "[VLM ANALYSIS FAILED] "
        "analysis budget exhausted without "
        "a valid scene submission",
        flush=True,
    )

    return {
        "wall_candidates": [],
        "ui_candidates": [],
        "important_objects": [],
        "goal_scores": {
            "reach_object": {
                "confidence": 0.0,
                "target_ids": [],
                "evidence": "analysis unavailable",
            },
            "match_pattern": {
                "confidence": 0.0,
                "target_ids": [],
                "evidence": "analysis unavailable",
            },
            "collect_objects": {
                "confidence": 0.0,
                "target_ids": [],
                "evidence": "analysis unavailable",
            },
            "activate_object": {
                "confidence": 0.0,
                "target_ids": [],
                "evidence": "analysis unavailable",
            },
            "move_into_region": {
                "confidence": 0.0,
                "target_ids": [],
                "evidence": "analysis unavailable",
            },
            "transform_shape": {
                "confidence": 0.0,
                "target_ids": [],
                "evidence": "analysis unavailable",
            },
            "unknown": {
                "confidence": 1.0,
                "target_ids": [],
                "evidence": (
                    "VLM failed to produce a valid "
                    "scene analysis"
                ),
            },
        },
    }

class MyAgentCore:
    MAX_ACTIONS = 20
    MAX_STALLED_ACTIONS = 6

    def __init__(self, episode_id=0):
        self.tracker = ObjectTracker(episode_id)
        self.previous_state = None
        self.previous_action = None
        self.action_effects = {}
        self.controlled_object_id = None
        self.action_index = 0
        self.action_vectors = {}
        self.scene_analysis = None
        self.action_components = {}
        self.controlled_component_ids = set()
        self.mode = "DISCOVER"
        self.previous_frame = None
        self.traversable_color_evidence = defaultdict(int)
        self.act_steps = 0
        self.act_steps_since_analysis = 0
        self.actions_without_progress = 0
        self.best_goal_distances = {}
        self.evidence_revision = 0
        self.goal_experiments = {}
        self.pending_interaction = None
        self.mechanics_evidence = {}
        self.needs_reanalysis = False
        self.controlled_track_ids = set()
        self.goal_target_tracks = {}
        self.failed_moves = set()
        self.last_action_state = None
        self.no_goal_steps = 0
        # Recent controlled-player positions for loop detection.
        self.recent_controlled_states = deque(maxlen=8)
        # Persistent visit counts for exploration.
        self.state_visit_counts = defaultdict(int)
        # Why the next VLM analysis is being requested.
        self.reanalysis_reason = None
        # Currently tested goal.
        self.current_goal = None
        self.current_goal_steps = 0
        # Learned navigation graph.
        self.transition_graph = defaultdict(dict)
        # Actions that have actually been attempted
        # from each controlled-player state.
        self.tested_actions = defaultdict(set)
        # State/action pairs that are known to be blocked.
        self.blocked_actions = set()
        

    def infer_controlled_entity(self):
        if len(self.action_components) < 4:
            return

        component_sets = list(
            self.action_components.values()
        )

        controlled = set.intersection(
            *component_sets
        )

        self.controlled_track_ids = controlled

        print(
            "CONTROLLED ENTITY:",
            sorted(controlled)
        )

    def learn_traversable_colors(
        self,
        previous_frame,
        transition,):
        groups = group_movements(
            transition.moved_objects
        )

        if not groups:
            return

        # Use the dominant movement group,
        # same as your action-learning logic.
        dominant_delta, movements = max(
            groups.items(),
            key=lambda item: len(item[1]),
        )

        # These are the pixels belonging to the
        # entity BEFORE the successful movement.
        controlled_pixels_before = set()

        for movement in movements:
            controlled_pixels_before |= (
                movement.before.pixels
            )

        colors = destination_colors(
            previous_frame,
            controlled_pixels_before,
            dominant_delta,
        )

        for color in colors:
            self.traversable_color_evidence[
                color
            ] += 1

        print(
            f"TRAVERSABLE EVIDENCE: "
            f"action={transition.action}, "
            f"delta={dominant_delta}, "
            f"colors={sorted(colors)}"
        )

        print(
            "TRAVERSABLE COLOR COUNTS:",
            dict(self.traversable_color_evidence),
        )

    def resolve_controlled_ids(self, objects):
        self.controlled_component_ids = set(resolve_goal_target_ids(
            objects, self.controlled_track_ids
        ))

    def clear_navigation_memory(
        self,):
        self.failed_moves.clear()
        self.last_action_state = None
        self.transition_graph.clear()
        self.tested_actions.clear()
        self.blocked_actions.clear()
        self.state_visit_counts.clear()
        self.recent_controlled_states.clear()

        print(
            "[NAV] failed-move memory cleared",
            flush=True,
        )

    def request_reanalysis(
        self,
        reason,
        reject_current_goal=False,
        record_goal_outcome=True,):
        print(
            "[REANALYZE REQUEST]",
            reason,
            flush=True,
        )

        if (
            record_goal_outcome
            and self.current_goal is not None
            and not self.needs_reanalysis
        ):
            self.record_goal_outcome(reason, reject_current_goal)

        if reject_current_goal and self.current_goal is not None:
            feedback = {
                "goal": self.current_goal,
                "reason": reason,
            }

            print(
                "[GOAL REJECTED]",
                feedback,
                flush=True,
            )

        self.reanalysis_reason = reason
        self.needs_reanalysis = True

    @staticmethod
    def goal_interaction(goal_type):
        # These labels currently execute the same movement/overlap planner.
        # Renaming a hypothesis must not bypass an unsuccessful experiment.
        if goal_type in {"reach_object", "move_into_region", "activate_object", "collect_objects"}:
            return "move_to_target"
        return goal_type

    def record_goal_outcome(self, reason, stalled):
        goal = self.current_goal
        interaction = self.goal_interaction(goal["type"])
        # A target can be a multi-piece marker.  Its members describe one
        # interaction, so remember the outcome for the group rather than
        # turning one visit into a separate failure for every piece.
        target_tracks = tuple(sorted(set(goal["target_tracks"]), key=repr))
        key = (interaction, target_tracks)
        record = self.goal_experiments.setdefault(key, {
            "interaction": interaction,
            "target_tracks": list(target_tracks),
            "goal_types": [],
            "stalled_count": 0,
            "inconclusive_count": 0,
            "last_stalled_revision": None,
        })
        if goal["type"] not in record["goal_types"]:
            record["goal_types"].append(goal["type"])
        record["stalled_count" if stalled else "inconclusive_count"] += 1
        if stalled:
            record["last_stalled_revision"] = self.evidence_revision
        record["last_outcome"] = "stalled_experiment" if stalled else "inconclusive"
        record["last_reason"] = reason

    def goal_memory_penalties(self):
        penalties = {}
        for goal_type, tracks in self.goal_target_tracks.items():
            if not tracks or not resolve_goal_target_ids(self.previous_state or [], tracks):
                penalties[goal_type] = None
                continue
            key = (
                self.goal_interaction(goal_type),
                tuple(sorted(set(tracks), key=repr)),
            )
            records = [self.goal_experiments[key]] if key in self.goal_experiments else []
            if any(record["last_stalled_revision"] == self.evidence_revision for record in records):
                penalties[goal_type] = None
            elif records:
                penalties[goal_type] = max(
                    min(0.4, 0.1 * record["stalled_count"] + 0.05 * record["inconclusive_count"])
                    for record in records
                )
        return penalties

    def cumulative_goal_feedback(self):
        # One entry per target group/interaction, retaining counts rather than an
        # unbounded event transcript or a sliding window that forgets failures.
        return [
            dict(record, retry_requires_new_evidence=(
                record["last_stalled_revision"] == self.evidence_revision
            ))
            for _, record in sorted(self.goal_experiments.items())
        ]

    @staticmethod
    def snapshot_track(obj):
        return {
            "color": obj.color,
            "shape_hash": obj.shape_hash,
            "pixel_count": len(obj.pixels),
            "bbox": list(obj.bbox),
        }

    def begin_interaction_observation(self, target_region_ids, planned_action=None):
        """Save the evidence needed to explain the result of one planned move."""
        if self.current_goal is None or self.previous_state is None:
            return None

        ui_ids = set((self.scene_analysis or {}).get("ui_candidates", []))
        ui_tracks = {
            obj.track_id
            for obj in self.previous_state
            if obj.id in ui_ids and obj.track_id is not None
        }
        snapshots = {
            obj.track_id: self.snapshot_track(obj)
            for obj in self.previous_state
            if (
                obj.track_id is not None
                and obj.track_id not in self.controlled_track_ids
            )
        }
        source_controlled_pixels = frozenset(get_controlled_pixels(
            self.previous_state, self.controlled_component_ids
        ))
        planned_action = planned_action or self.previous_action
        planned_delta = self.action_vectors.get(planned_action)
        predicted_destination_pixels = frozenset(
            translate_pixels(source_controlled_pixels, planned_delta)
        ) if source_controlled_pixels and planned_delta is not None else frozenset()
        target_region_pixels = frozenset(
            get_object_pixels(self.previous_state, target_region_ids)
        )
        return {
            "goal_type": self.current_goal["type"],
            "target_tracks": tuple(self.current_goal["target_tracks"]),
            "target_region_pixels": target_region_pixels,
            "planned_action": planned_action,
            "predicted_destination_pixels": predicted_destination_pixels,
            "predicted_contact": bool(
                predicted_destination_pixels & target_region_pixels
            ),
            "ui_tracks": ui_tracks,
            "snapshots": snapshots,
        }

    def record_mechanics_observation(self, observation):
        """Aggregate contact effects without calling a one-off animation causal."""
        effect_tracks = tuple(sorted(
            (change["track_id"], change["kind"], change["area"])
            for change in (
                observation["ui_changes"]
                + observation["gameplay_changes"]
            )
        ))
        key = (
            tuple(observation["target_tracks"]),
            tuple(observation["target_not_visible"]),
            effect_tracks,
        )
        record = self.mechanics_evidence.setdefault(key, {
            "goal_type": observation["goal_type"],
            "target_tracks": list(observation["target_tracks"]),
            "target_not_visible": list(observation["target_not_visible"]),
            "ui_changes": observation["ui_changes"],
            "gameplay_changes": observation["gameplay_changes"],
            "contact_count": 0,
            "no_effect_count": 0,
        })
        record["contact_count"] += 1
        if not effect_tracks and not observation["target_not_visible"]:
            record["no_effect_count"] += 1

        if record["contact_count"] >= 2:
            confidence = "repeated"
        elif observation["target_not_visible"]:
            confidence = "strong_single_observation"
        else:
            confidence = "observed_once"
        record["association_confidence"] = confidence
        observation["association_confidence"] = confidence
        return record

    def cumulative_mechanics_feedback(self):
        return [
            dict(record)
            for _, record in sorted(
                self.mechanics_evidence.items(),
                key=lambda item: repr(item[0]),
            )
        ]

    def observe_pending_interaction(self, objects, movement_succeeded=False):
        pending = self.pending_interaction
        self.pending_interaction = None
        if pending is None:
            return None

        controlled_pixels = get_controlled_pixels(
            objects, self.controlled_component_ids
        )
        actual_contact = bool(
            controlled_pixels & pending["target_region_pixels"]
        )
        # Final contact can consume, recolor, or merge the target before the
        # tracker can resolve its old IDs.  A verified movement into the
        # precomputed destination is therefore contact evidence as well.
        predicted_contact = bool(
            movement_succeeded and pending.get("predicted_contact")
        )
        if not actual_contact and not predicted_contact:
            return None

        after = {
            obj.track_id: obj
            for obj in objects
            if (
                obj.track_id is not None
                and obj.track_id not in self.controlled_track_ids
            )
        }
        changes = []
        for track_id, before in pending["snapshots"].items():
            current = after.get(track_id)
            area = "ui" if track_id in pending["ui_tracks"] else "gameplay"
            if current is None:
                changes.append({
                    "track_id": track_id,
                    "kind": "not_visible",
                    "area": area,
                })
                continue

            current_snapshot = self.snapshot_track(current)
            fields = [
                name for name in ("color", "shape_hash", "pixel_count", "bbox")
                if before[name] != current_snapshot[name]
            ]
            if fields:
                changes.append({
                    "track_id": track_id,
                    "kind": "changed",
                    "area": area,
                    "fields": fields,
                })

        # A split/merge replacement has a new track ID, but its lineage still
        # makes it an observable change to the original object.
        known_tracks = set(pending["snapshots"])
        for obj in after.values():
            parents = set(obj.parent_track_ids) & known_tracks
            if parents:
                changes.append({
                    "track_id": obj.track_id,
                    "kind": "lineage_changed",
                    "area": (
                        "ui" if parents & pending["ui_tracks"] else "gameplay"
                    ),
                    "parent_tracks": sorted(parents),
                })

        target_not_visible = sorted(
            track for track in pending["target_tracks"]
            if track not in after
        )
        observation = {
            "goal_type": pending["goal_type"],
            "target_tracks": list(pending["target_tracks"]),
            "action": (pending.get("planned_action") or self.previous_action).name,
            "contacted": True,
            "contact_method": "actual" if actual_contact else "predicted",
            "target_not_visible": target_not_visible,
            "ui_changes": [change for change in changes if change["area"] == "ui"],
            "gameplay_changes": [
                change for change in changes if change["area"] == "gameplay"
            ],
        }
        record = self.record_mechanics_observation(observation)
        print("[INTERACTION EFFECT]", json.dumps(observation), flush=True)

        if (
            observation["target_not_visible"]
            or observation["ui_changes"]
            or observation["gameplay_changes"]
        ):
            self.request_reanalysis(
                reason=(
                    "target contact produced an observed effect: "
                    f"ui={len(observation['ui_changes'])}, "
                    f"gameplay={len(observation['gameplay_changes'])}, "
                    f"confidence={record['association_confidence']}"
                ),
                reject_current_goal=False,
                record_goal_outcome=False,
            )

        return observation

    def learn(self, action, transition):
        if action is None:
            return
        groups = group_movements(
            transition.moved_objects
        )
        if not groups:
            return

        dominant_delta, movements = max(
            groups.items(),
            key=lambda item: len(item[1]),
        )
        self.action_vectors[action] = dominant_delta
        moving_ids = {
            movement.before.track_id
            for movement in movements
            if movement.before.track_id is not None
        }

        self.action_components[action] = moving_ids
        print(
            f"LEARNED: {action.name} "
            f"-> {dominant_delta}, "
            f"components={sorted(moving_ids)}"
        )
        self.infer_controlled_entity()

    def observe(self, frame):
        frame = np.asarray(frame)
        new_evidence = False
        observed_action = False
        expected_move_seen = False

        previous_controlled_ids = set(self.controlled_component_ids)

        objects = self.tracker.update(
            extract_objects(frame), self.controlled_track_ids,
            self.action_vectors.get(self.previous_action),
        )

        if (
            self.previous_state is not None
            and self.previous_frame is not None
            and self.previous_action is not None
        ):
            transition = describe_transition(
                self.previous_state,
                self.previous_action,
                objects,
            )
            if self.mode == "DISCOVER":
                self.learn(
                    self.previous_action,
                    transition,
                )
            elif self.mode == "ACT":
                observed_action = self.last_action_state is not None
                # reanalysis if move was not expected
                expected_delta = (
                    self.action_vectors.get(
                        self.previous_action
                    )
                )

                controlled_movements = [
                    movement
                    for movement
                    in transition.moved_objects
                    if (movement.before.id in previous_controlled_ids and 
                        movement.delta == expected_delta)
                ]

                expected_move_seen = bool(controlled_movements)

                print(
                    "[ACT RESULT]",
                    f"action={self.previous_action.name}",
                    f"expected={expected_delta}",
                    f"success={expected_move_seen}",
                    flush=True,
                )
                # The state from which the previous action was issued.
                source_state = self.last_action_state

                if source_state and resolve_goal_target_ids(objects, self.controlled_track_ids):

                    # A newly tested move is useful even when it is blocked.
                    new_evidence = (
                        self.previous_action not in self.tested_actions[source_state]
                    )

                    # We have now experimentally tested this action
                    # from this state.
                    self.tested_actions[source_state].add(
                        self.previous_action
                    )
                    if expected_move_seen:

                        # Build the resulting controlled-player state
                        # from the movement result.
                        destination_pixels = set()

                        for movement in controlled_movements:
                            destination_pixels |= (
                                movement.after.pixels
                            )

                        destination_state = frozenset(
                            destination_pixels
                        )

                        if destination_state:
                            new_evidence |= (
                                self.transition_graph[source_state].get(self.previous_action)
                                != destination_state
                            )
                            self.transition_graph[
                                source_state
                            ][self.previous_action] = (
                                destination_state
                            )

                            print(
                                "[TRANSITION LEARNED]",
                                f"{self.previous_action.name}:",
                                f"{len(source_state)}px",
                                "->",
                                f"{len(destination_state)}px",
                                flush=True,
                            )

                    elif not any(old.track_id in self.controlled_track_ids
                                 for old, _ in transition.changed_objects):

                        self.failed_moves.add(
                            (
                                source_state,
                                self.previous_action,
                            )
                        )

                        self.blocked_actions.add(
                            (
                                source_state,
                                self.previous_action,
                            )
                        )

                        print(
                            "[BLOCKED TRANSITION]",
                            self.previous_action.name,
                            flush=True,
                        )

            self.learn_traversable_colors(
                self.previous_frame,
                transition,
            )

        if self.tracker.events:
            print("[TRACK EVENTS]", self.tracker.events, flush=True)
        # Transfer the controlled role only across lineage wholly belonging to it.
        descendants = [obj for obj in objects if obj.parent_track_ids
                       and set(obj.parent_track_ids) <= self.controlled_track_ids
                       and obj.tracking_status == "lineage"]
        replaced = {parent for obj in descendants for parent in obj.parent_track_ids}
        if replaced:
            self.controlled_track_ids = (self.controlled_track_ids - replaced) | {
                obj.track_id for obj in descendants
            }
        if self.controlled_track_ids:
            self.resolve_controlled_ids(objects)

        interaction_observation = self.observe_pending_interaction(
            objects, movement_succeeded=expected_move_seen
        )
        if interaction_observation is not None:
            new_evidence = True

        if (self.mode == "ACT" and self.current_goal is not None
                and not self.needs_reanalysis
                and not (
                    interaction_observation
                    and interaction_observation["contacted"]
                )
                and not resolve_goal_target_ids(objects, self.current_goal["target_tracks"])):
            self.request_reanalysis(
                "target identity unresolved or split/merged; collection is not established",
                reject_current_goal=False,
            )

        if self.mode == "ACT":

            current_pixels = frozenset(
                get_controlled_pixels(
                    objects,
                    self.controlled_component_ids,
                )
            )

            if observed_action:
                target_tracks = {track for tracks in self.goal_target_tracks.values() for track in tracks}
                target_changed = any(
                    old.track_id in target_tracks for old, _ in transition.changed_objects
                ) or any(set(obj.parent_track_ids) & target_tracks
                         for obj in objects if obj.tracking_status == "lineage")
                controlled_changed = any(
                    obj.tracking_status == "lineage" for obj in descendants
                ) or any(
                    obj.track_id in self.controlled_track_ids
                    and old.shape_hash != obj.shape_hash
                    for old, obj in (transition.changed_objects if self.previous_state is not None else [])
                )
                if controlled_changed or target_changed:
                    self.clear_navigation_memory()
                    new_evidence = True
                new_evidence |= bool(current_pixels) and (
                    self.state_visit_counts[current_pixels] == 0
                )
                new_evidence = self.record_navigation_progress(
                    objects, current_pixels, new_evidence
                )

            if current_pixels:

                self.state_visit_counts[
                    current_pixels
                ] += 1

                if observed_action:
                    self.check_navigation_loop(current_pixels, new_evidence)
        self.previous_state = objects
        self.previous_frame = frame.copy()

        return objects

    def check_navigation_loop(self, current_pixels, new_evidence):
        # New experiments/progress break a no-progress sequence, including
        # testing a different blocked action while remaining in place.
        if new_evidence:
            self.recent_controlled_states.clear()

        recent = self.recent_controlled_states
        reason = None
        reject_goal = False
        if not new_evidence and not self.needs_reanalysis:
            if (
                len(recent) >= 2
                and current_pixels == recent[-2]
                and current_pixels != recent[-1]
                and self.actions_without_progress >= 2
            ):
                reason = "two-state movement loop detected"
                reject_goal = self.current_goal is not None
            elif (
                len(recent) >= 2
                and current_pixels == recent[-1] == recent[-2]
                and self.actions_without_progress >= 3
            ):
                # Repeated blocked moves do not disprove the target hypothesis.
                reason = "repeated stationary actions without new evidence"
            elif (
                recent.count(current_pixels) >= 2
                and self.actions_without_progress >= 3
            ):
                reason = "controlled state repeated within recent no-progress window"
                reject_goal = self.current_goal is not None

        recent.append(current_pixels)
        if reason is not None:
            self.request_reanalysis(reason, reject_current_goal=reject_goal)

    def record_navigation_progress(self, objects, current_pixels, new_evidence):
        # Only a new best distance counts; approaching after moving away
        # must not repeatedly reset the stall counter.
        if self.current_goal is not None and current_pixels:
            tracks = self.current_goal["target_tracks"]
            key = (self.current_goal["type"], tuple(sorted(tracks)))
            target_ids = resolve_goal_target_ids(objects, tracks)
            target_pixels = get_object_pixels(objects, target_ids)
            distance = pixel_distance(current_pixels, target_pixels)
            best = self.best_goal_distances.get(key, float("inf"))
            if distance < best:
                self.best_goal_distances[key] = distance
                new_evidence = True

        if new_evidence:
            self.evidence_revision += 1
            self.actions_without_progress = 0
        else:
            self.actions_without_progress += 1

        print("[PROGRESS]", f"new_evidence={new_evidence}",
              f"stalled_actions={self.actions_without_progress}", flush=True)
        return new_evidence

    def commit_action(
        self,
        action,
        source,
        interaction_context=None,):
        controlled_pixels = (
            get_controlled_pixels(
                self.previous_state,
                self.controlled_component_ids,
            )
        )

        self.last_action_state = (
            frozenset(
                controlled_pixels
            )
        )

        print(
            f"[ACT] source={source}, "
            f"action={action.name}, "
            f"state_pixels="
            f"{len(self.last_action_state)}",
            flush=True,
        )

        self.pending_interaction = interaction_context
        self.previous_action = action
        self.act_steps_since_analysis += 1

        return action

    def choose_action(self,
        frame: np.ndarray,
        available_actions: list[int | arcengine.GameAction],) -> arcengine.GameAction:

        legal_actions = [
            value if isinstance(value, arcengine.GameAction)
            else arcengine.GameAction.from_id(value)
            for value in available_actions
        ]
        if not legal_actions:
            raise ValueError("No available actions to choose from")
        fallback_action = next(
            (action for action in legal_actions if action != arcengine.GameAction.RESET),
            legal_actions[0],
        )

        objects = self.observe(frame)
        if self.controlled_track_ids and not self.controlled_component_ids:
            print("[TRACK] controlled entity unresolved; taking a new observation", flush=True)
            self.last_action_state = None
            action = legal_actions[self.action_index % len(legal_actions)]
            self.action_index += 1
            self.previous_action = action
            if all(track in self.tracker.retired or self.tracker.missing.get(track, 0) > self.tracker.MAX_MISSING
                   for track in self.controlled_track_ids):
                # Relearn the controlled role without pretending a new object is the old one.
                self.controlled_track_ids.clear()
                self.action_vectors.clear()
                self.action_components.clear()
                self.current_goal = None
                self.mode = "DISCOVER"
                self.clear_navigation_memory()
            return action
        # Keep learned controls intact; availability can change between frames.
        legal_action_vectors = {
            action: delta
            for action, delta in self.action_vectors.items()
            if action in legal_actions
        }
        # Reconsider only after consecutive actions without useful evidence.
        if (
            self.mode == "ACT"
            and not self.needs_reanalysis
            and self.actions_without_progress >= self.MAX_STALLED_ACTIONS
        ):
            self.request_reanalysis(
                reason=(
                    "six consecutive actions without new evidence or goal progress; "
                    "target not reached, hypothesis remains untested or inconclusive"
                ),
                reject_current_goal=False,
            )

        # Exploration without an actionable goal
        # should also be temporary.
        if (
            self.mode == "ACT"
            and self.no_goal_steps >= 3
            and self.actions_without_progress >= 3
            and not self.needs_reanalysis
        ):
            self.request_reanalysis(
                reason=(
                    "exploration without an actionable goal has produced "
                    "no new evidence for three actions"
                ),
                reject_current_goal=False,
            )
        # -------------------------
        # DISCOVER -> ANALYZE
        # -------------------------
        if (
            self.mode == "DISCOVER"
            and len(self.action_vectors) >= 4
            and self.controlled_component_ids
        ):
            print("[MODE] DISCOVER -> ANALYZE",flush=True,)
            self.controlled_track_ids = {
                obj.track_id
                for obj in objects if obj.id
                in self.controlled_component_ids
            }

            print(
                "[CONTROLLED TRACKS]",
                self.controlled_track_ids,
                flush=True,
            )
            self.mode = "ANALYZE"
        # -------------------------
        # ACT -> ANALYZE
        # -------------------------
        if (
            self.mode == "ACT"
            and self.needs_reanalysis
        ):
            print("[MODE] ACT -> ANALYZE",flush=True,)
            self.mode = "ANALYZE"
        # -------------------------
        # ANALYZE
        # -------------------------
        if self.mode == "ANALYZE":
            print(
                "[ANALYSIS CONTEXT]",
                "reason=",
                self.reanalysis_reason,
                "feedback=",
                json.dumps(
                    self.cumulative_goal_feedback(),
                    indent=2,
                ),
                "mechanics=",
                json.dumps(
                    self.cumulative_mechanics_feedback(),
                    indent=2,
                ),
                flush=True,
            )
            self.scene_analysis = analyze_scene_vlm(
                frame=frame,
                objects=objects,
                action_vectors=self.action_vectors,
                controlled_components=self.controlled_component_ids,
                traversable_color_evidence=self.traversable_color_evidence,
                previous_feedback=self.cumulative_goal_feedback(),
                reanalysis_reason=self.reanalysis_reason,
                tracking_events=self.tracker.events,
                causal_observations=self.cumulative_mechanics_feedback(),
            )

            self.goal_target_tracks = (capture_goal_target_tracks(self.scene_analysis,objects,))
            print("[TARGET TRACKS]",self.goal_target_tracks,flush=True,)

            print("\n=== VLM SCENE ANALYSIS ===")
            print(json.dumps(self.scene_analysis,indent=2,))

            self.reanalysis_reason = None
            self.recent_controlled_states.clear()
            self.no_goal_steps = 0
            self.needs_reanalysis = False
            self.act_steps_since_analysis = 0
            self.actions_without_progress = 0
            self.mode = "ACT"
        # -------------------------
        # ACT
        # -------------------------
        if self.mode == "ACT":

            goal = select_primary_goal(
                self.scene_analysis,
                goal_penalties=self.goal_memory_penalties(),
            )
          
            if goal is None:
                self.no_goal_steps += 1
                self.current_goal = None
                self.current_goal_steps = 0

                print(
                    "[ACT] no actionable goal, "
                    f"exploration_step={self.no_goal_steps}",
                    flush=True,
                )

            else:
                self.no_goal_steps = 0

                target_tracks = (self.goal_target_tracks.get(goal["type"],[],))

                goal_identity = {
                    "type": goal["type"],
                    "target_tracks": sorted(
                        set(target_tracks)
                    ),
                }

                if goal_identity != self.current_goal:

                    print(
                        "[GOAL TEST START]",
                        goal_identity,
                        flush=True,
                    )

                    self.current_goal = goal_identity
                    self.current_goal_steps = 0

                else:
                    self.current_goal_steps += 1
            print(
                "[ACT] selected goal:",
                goal,
                flush=True,
            )

            SUPPORTED_GOALS = {
                "reach_object",
                "move_into_region",
                "activate_object",
                "collect_objects"
            }

            # -------------------------
            # 1. Try goal-directed plan
            # -------------------------
            if goal is not None and goal["type"] in SUPPORTED_GOALS:
                # resolve target ids
                target_tracks = (self.goal_target_tracks.get(goal["type"],[],))

                target_ids = (
                    resolve_goal_target_ids(
                        self.previous_state,
                        target_tracks,
                    )
                )

                print(
                    "[TARGET RESOLVED]",
                    f"type={goal['type']}",
                    f"tracks={target_tracks}",
                    f"current_ids={target_ids}",
                    flush=True,
                )
                
                if not target_ids:
                    print(
                        "[TARGET] target disappeared "
                        "or cannot be resolved",
                        flush=True,
                    )

                    self.needs_reanalysis = True

                else:
                    target_region_ids = expand_goal_target_region(
                        objects=self.previous_state,
                        target_ids=target_ids,
                        controlled_ids=self.controlled_component_ids,
                        ui_ids=self.scene_analysis["ui_candidates"],
                        wall_ids=self.scene_analysis["wall_candidates"],
                        traversable_color_evidence=self.traversable_color_evidence,
                    )

                    print(
                        "[TARGET REGION]",
                        f"anchor_ids={target_ids}",
                        f"contact_ids={target_region_ids}",
                        flush=True,
                    )

                    action = choose_goal_action_bfs(
                        frame=np.asarray(frame),
                        objects=self.previous_state,
                        controlled_ids=(self.controlled_component_ids),
                        target_ids=target_region_ids,
                        action_vectors=legal_action_vectors,
                        failed_moves=(self.failed_moves),
                        traversable_color_evidence=(self.traversable_color_evidence),
                    )
                    
                    if action is not None:
                        return self.commit_action(
                            action,
                            "goal_planner",
                            interaction_context=(
                                self.begin_interaction_observation(
                                    target_region_ids, action
                                )
                            ),
                        )
                    
                    if action is None:
                        print(
                            "[BFS] no verified route to current target; "
                            "keeping current goal and exploring",
                            flush=True,
                        )
                        action = choose_frontier_action(
                            objects=self.previous_state,
                            controlled_ids=(self.controlled_component_ids),
                            action_vectors=legal_action_vectors,
                            transition_graph=self.transition_graph,
                            tested_actions=self.tested_actions,
                            blocked_actions=self.blocked_actions,
                            previous_action=self.previous_action
                        )
                        if action is None:
                            action = choose_exploration_action(
                                frame=np.asarray(frame),
                                objects=self.previous_state,
                                controlled_ids=(self.controlled_component_ids),
                                action_vectors=legal_action_vectors,
                                scene_analysis=(self.scene_analysis),
                                traversable_color_evidence=(self.traversable_color_evidence),
                                state_visit_counts=self.state_visit_counts,
                            )

                    if action is None:
                        print("No action chosen, using available fallback action")
                        action = fallback_action
                    if action is not None:
                        return self.commit_action(action,"exploration_after_bfs_failure",)

            # --------------------------------
            # 2. Goal understood but unsupported
            # --------------------------------
            if (goal is not None and goal["type"] not in SUPPORTED_GOALS):
                print(
                    "[ACT] goal understood but "
                    "planner does not support it:",
                    goal["type"],
                    flush=True,
                )

            # -------------------------
            # 3. Exploration fallback
            # -------------------------
            action = choose_frontier_action(
                objects=self.previous_state,
                controlled_ids=(self.controlled_component_ids),
                action_vectors=legal_action_vectors,
                transition_graph=self.transition_graph,
                tested_actions=self.tested_actions,
                blocked_actions=self.blocked_actions,
                previous_action=self.previous_action,
            )
            if action is None:
                action = choose_exploration_action(
                    frame=np.asarray(frame),
                    objects=self.previous_state,
                    controlled_ids=(self.controlled_component_ids),
                    action_vectors=legal_action_vectors,
                    scene_analysis=(self.scene_analysis),
                    traversable_color_evidence=(self.traversable_color_evidence),
                    state_visit_counts=self.state_visit_counts,
                )

            if action is not None: 
                return self.commit_action(action,"exploration",)
            
            # -------------------------
            # 4. Absolute fallback
            # -------------------------
            print("[ACT] no justified action found", flush=True,)

            return self.commit_action(fallback_action,"absolute_fallback",)
        
        if self.mode == "DISCOVER":

            actions = [
                arcengine.GameAction.ACTION1,
                arcengine.GameAction.ACTION2,
                arcengine.GameAction.ACTION3,
                arcengine.GameAction.ACTION4,
            ]

            # Only use currently legal actions.
            actions = [
                action
                for action in actions
                if action in legal_actions
            ]

            if not actions:
                self.previous_action = fallback_action
                return fallback_action

            action = actions[
                self.action_index % len(actions)
            ]

            self.action_index += 1
            self.previous_action = action

            return action
        
@dataclass
class MyAgentSolver(Solver):
    label: str = "MyAgent2"

    max_actions_per_game: int = 100

    async def _run_games(
        self,
        games: list[taaf.game.Game],
    ) -> None:

        try:
            await asyncio.gather(
                *(
                    self._play_one(game)
                    for game in games
                )
            )

        except asyncio.CancelledError:

            for game in games:
                run = game.game_run

                if (
                    run is not None
                    and run.final_score is None
                ):
                    game.finish_game()

            raise

    async def _play_one(
        self,
        game: taaf.game.Game,
    ) -> None:

        agent = MyAgentCore()
        actions_taken = 0
        episode_id = 0
        previous_level = None

        try:
            while True:

                # Required so benchmark cancellation can run.
                await asyncio.sleep(0)

                run = game.game_run

                if (
                    run is None
                    or run.state != "playing"
                ):
                    break

                if (
                    actions_taken
                    >= self.max_actions_per_game
                ):
                    break

                state = game.current_state
                level = getattr(state.raw, "levels_completed", None)
                if previous_level is not None and level is not None and level != previous_level:
                    episode_id += 1
                    agent = MyAgentCore(episode_id)
                previous_level = level

                # Handle engine GAME_OVER.
                if (
                    state.raw.state
                    == arcengine.GameState.GAME_OVER
                ):
                    action = (
                        arcengine.GameAction.RESET
                    )

                else:

                    frame = np.asarray(
                        state.frame.data
                    )

                    action = agent.choose_action(
                        frame=frame,
                        available_actions=list(
                            state.available_actions
                        ),
                    )

                action_input = (
                    arcengine.ActionInput(
                        id=action,
                        data={},
                    )
                )

                game.execute_action(
                    action_input,
                )

                actions_taken += 1
                if action == arcengine.GameAction.RESET:
                    episode_id += 1
                    agent = MyAgentCore(episode_id)
                    previous_level = None

            if (
                game.game_run is not None
                and game.game_run.final_score is None
            ):
                game.finish_game()

        except asyncio.CancelledError:

            if (
                game.game_run is not None
                and game.game_run.final_score is None
            ):
                game.finish_game()

            raise
        except Exception as exc:
            game_name = (
                getattr(game, "game_id", None)
                or getattr(game, "env_name", None)
                or "unknown"
            )

            print(
                f"\n[MYAGENT ERROR] {game_name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            traceback.print_exc()

            run = game.game_run

            if run is not None:
                run.solver_note = (
                    f"{type(exc).__name__}: {exc}"
                )

                if run.final_score is None:
                    with contextlib.suppress(Exception):
                        game.finish_game()
