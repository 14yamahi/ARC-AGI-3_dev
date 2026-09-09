from dataclasses import dataclass
from collections import deque, defaultdict
import numpy as np
from typing import Any
import base64
import io
import json
from openai import OpenAI
from PIL import Image
from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState
import os

@dataclass
class GameObject:
    id: int
    color: int
    bbox: tuple
    center: tuple
    pixels: set
    width: int
    height: int

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
class SceneModel:
    controlled_entity: set[int]

    likely_walls: list[int]
    likely_floor_colors: set[int]

    interesting_objects: list[int]

    spatial_relations: list

    goal_hypotheses: list

@dataclass
class ActionPrediction:
    action: GameAction

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

LOCAL_BASE_URL = os.getenv(
    "ARC_LOCAL_BASE_URL",
    "http://127.0.0.1:1234/v1",
)

LOCAL_MODEL = os.getenv(
    "ARC_LOCAL_MODEL",
    "vrfai/Qwen3.6-27B-FP8",
)

client = OpenAI(
    base_url=LOCAL_BASE_URL,
    api_key="local",
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
def frame_to_data_url(
    frame,
    save_path=None
):
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
    image = image.resize(
        (w * 8, h * 8),
        Image.Resampling.NEAREST,
    )

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
def extract_objects(frame) -> list[GameObject]:
    frame = np.asarray(frame)

    # You may need to change this depending on ARC-AGI-3 frames.
    background = 4

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

            if color == background:
                continue

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

            obj = GameObject(
                id=object_id,
                color=color,
                bbox=(min_y, min_x, max_y, max_x),
                center=(
                    sum(ys) / len(ys),
                    sum(xs) / len(xs),
                ),
                pixels=pixels,
                width=max_x - min_x + 1,
                height=max_y - min_y + 1,
            )

            objects.append(obj)
            object_id += 1

    return objects

def group_movements(movements):

    groups = defaultdict(list)

    for movement in movements:
        groups[movement.delta].append(movement)

    return groups

# link objects that moved to the same id
def match_objects(before, after):
    MAX_MATCH_DISTANCE = 15

    matches = []
    unmatched_after = set(range(len(after)))

    for old in before:

        candidates = []

        for i in unmatched_after:
            new = after[i]

            if old.color != new.color:
                continue

            if old.width != new.width:
                continue

            if old.height != new.height:
                continue

            distance = (
                abs(old.center[0] - new.center[0])
                + abs(old.center[1] - new.center[1])
            )

            candidates.append(
                (distance, i, new)
            )

        if not candidates:
            matches.append((old, None))
            continue

        distance, i, new = min(
            candidates,
            key=lambda x: x[0],
        )

        if distance > MAX_MATCH_DISTANCE:
            matches.append((old, None))
            continue

        # We found the corresponding object
        matches.append((old, new))

        # Do not allow another old object
        # to match this same new object
        unmatched_after.remove(i)

    appeared = [
        after[i]
        for i in unmatched_after
    ]

    return matches, appeared

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
            
        if old_shape == new_shape:
            # Same shape translated somewhere.

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
        else:
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
    objects: list[GameObject],
) -> list[SpatialRelation]:

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
    action_vectors,
) -> SceneModel:
    return None

def validate_scene_analysis(
    analysis,
    controlled_ids,
    objects,
    traversable_color_evidence,
):
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

    # Controlled entity should not be listed as
    # an interesting/target object.
    analysis["important_objects"] = [
        obj_id
        for obj_id in analysis["important_objects"]
        if obj_id not in controlled_ids
    ]

    # Remove controlled components from
    # every goal's target list.
    for result in (
        analysis["goal_scores"].values()
    ):
        original_targets = set(
            result["target_ids"]
        )

        cleaned_targets = (
            original_targets
            - controlled_ids
        )

        result["target_ids"] = sorted(
            cleaned_targets
        )

        # If the only proposed targets were the player,
        # invalidate this hypothesis.
        if (
            original_targets
            and not cleaned_targets
        ):
            result["confidence"] = 0.0

    return analysis

def build_composite_regions(
    relationships,
):
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
    controlled_ids,
):
    pixels = set()

    for obj in objects:
        if obj.id in controlled_ids:
            pixels |= obj.pixels

    return pixels

def translate_pixels(
    pixels,
    delta,
):
    dx, dy = delta

    return {
        (y + dy, x + dx)
        for y, x in pixels
    }

def destination_colors(
    frame,
    controlled_pixels,
    delta,
):
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
    min_confidence=0.5,
):
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

        candidates.append(
            (
                result["confidence"],
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
    object_ids,
):
    ids = set(object_ids)
    pixels = set()

    for obj in objects:
        if obj.id in ids:
            pixels |= obj.pixels

    return pixels

def pixel_distance(
    pixels_a,
    pixels_b,
):
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
    traversable_color_evidence,
):
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
):
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

        # Prefer safe movement, but give a small bonus
        # for exploring something not yet understood.
        score = (
            known_traversable
            + unknown * 0.25
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

def track_controlled_entity(
    self,
    transition,
):
    if not self.controlled_component_ids:
        return

    expected_delta = self.action_vectors.get(
        self.previous_action
    )

    if expected_delta is None:
        return

    current_ids = set()

    for movement in transition.moved_objects:

        if (
            movement.before.id
            in self.controlled_component_ids
            and movement.delta == expected_delta
        ):
            current_ids.add(
                movement.after.id
            )

    if current_ids:
        self.controlled_component_ids = current_ids

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
    },
    "required": [
        "wall_candidates",
        "important_objects",
        "goal_scores",
    ],
    "additionalProperties": False,
}

def analyze_scene_vlm(
    frame,
    objects,
    action_vectors,
    controlled_components,
    traversable_color_evidence,
):
    image_url = frame_to_data_url(frame, save_path="debug_frames/debug_scene.png",)

    object_description = "\n".join(
        (
            f"Object {obj.id}: "
            f"color={obj.color}, "
            f"bbox={obj.bbox}, "
            f"center={obj.center}, "
            f"size={len(obj.pixels)}"
        )
        for obj in objects
    )

    # check if objects are contained in other objects
    relationships = generate_containment_relationships(objects)
    relationship_description = "\n".join(
        f"Object {r.a} {r.relation} Object {r.b}"
        for r in relationships
    )
    regions = build_composite_regions(
        relationships
    )

    region_description = "\n".join(
        (
            f"Region: container Object {r.container_id}, "
            f"contents={r.content_ids}"
        )
        for r in regions
    )
    action_description = "\n".join(
        (
            f"{action.name}: {delta}"
        )
        for action, delta
        in action_vectors.items()
    )

    # traversable color
    if traversable_color_evidence:
        traversability_description = "\n".join(
            (
                f"Color {color}: "
                f"{count} successful traversal observations"
            )
            for color, count
            in sorted(
                traversable_color_evidence.items()
            )
        )
    else:
        traversability_description = (
            "No experimentally verified "
            "traversable colors yet."
        )
    prompt = f"""
You are analyzing an unknown interactive visual puzzle.

The game mechanics and objective are unknown.
Do not assume this is a navigation game, collection game,
pattern-matching game, or any other specific type of game.

========================
VERIFIED EXPERIMENTAL FACTS
========================

The agent experimentally observed these action effects:

{action_description}

Experimentally identified controllable components:

{sorted(controlled_components)}

These components moved consistently in direct response to the
agent's actions. Treat them as parts of the controllable entity.

Do not assume the controllable entity is necessarily a traditional
player/avatar.

========================
DETECTED COMPONENTS
========================

{object_description}

These components were deterministically extracted from the frame.

========================
SPATIAL RELATIONSHIPS
========================

{relationship_description}

IMPORTANT:
BBOX_INSIDE means only that one object's bounding box lies inside
another object's bounding box.

It does NOT prove that the outer object is a semantic container
or that the inner object is truly enclosed by it.

========================
COMPOSITE REGION CANDIDATES
========================

{region_description}

These regions are heuristic groupings derived from bounding-box
relationships. Treat them as structural clues, not verified objects.

========================
EXPERIMENTAL TERRAIN EVIDENCE
========================

{traversability_description}

A successful traversal observation means that the controllable
entity moved into cells of that color during an experimentally
observed action.

This is evidence that the color may represent traversable terrain
in the current game.

Do not treat a color with successful traversal evidence as
non-traversable solely because of its visual appearance.

This does not prove that every cell of that color is always
traversable; game mechanics may be context-dependent.

========================
COORDINATE CONVENTION
========================

Movement vectors use (dx, dy).

dx > 0 = RIGHT
dx < 0 = LEFT
dy > 0 = DOWN
dy < 0 = UP

Therefore:

(0, -5) = UP
(0, 5)  = DOWN
(-5, 0) = LEFT
(5, 0)  = RIGHT

========================
EVIDENCE PRIORITY
========================

When evidence conflicts, use this priority:

1. Experimentally verified action effects and controllable components
2. Deterministically detected component geometry
3. Computed spatial relationships and composite-region candidates
4. Visual interpretation of the image
5. General assumptions about how games usually work

Never override higher-priority evidence with a lower-priority guess.

========================
ANALYSIS TASK
========================

First identify:

1. wall_candidates
   - Objects that may behave as obstacles, barriers, boundaries,
     or non-traversable structures.

2. important_objects
   - Non-controlled objects that appear structurally unusual,
     interactive, target-like, reference-like, or otherwise relevant.

Then evaluate EVERY goal category below:

- reach_object
- match_pattern
- collect_objects
- activate_object
- move_into_region
- transform_shape
- unknown

For every category provide:

- confidence from 0.0 to 1.0
- relevant target object IDs
- concise evidence based on the current scene

Confidence measures strength of evidence, not certainty.
The values do not need to sum to 1.

Use confidence 0.0 when there is no meaningful evidence.

For unknown:
- use higher confidence when the current evidence does not strongly
  distinguish among the other goal categories.
- target_ids should normally be empty.

If a goal has no plausible target object, return an empty target_ids list.

========================
COLOR REASONING
========================

Color values are symbolic visual categories.

Do NOT assume a universal semantic meaning for a color
(for example, do not assume blue is water or black is a wall).

However, color is important evidence.

When inferring object roles, consider:

- Objects with the same color may share a material or semantic role.
- Large connected regions of one color may represent floor,
  terrain, background, walls, or boundaries.
- A color occupied by or successfully entered by the controllable
  entity is evidence that this color may be traversable.
- A color repeatedly bordering or blocking the controllable entity
  may be evidence that it is non-traversable.
- Small regions whose colors differ sharply from surrounding terrain
  may be interactive objects, markers, targets, or indicators.
- Repeated color patterns across structurally similar regions may be
  evidence for a pattern-matching or transformation objective.

Never infer a role from color alone.
Combine color with geometry, movement evidence, connectivity,
location, containment, and repeated structure.

========================
RULES
========================

- Never classify controlled components as walls, obstacles,
  collectibles, targets, or goal objects.

- Never include controlled component IDs in target_ids.

- Do not reinterpret experimentally verified action vectors.

- Do not infer a goal merely because an object is large,
  centrally located, brightly colored, or visually prominent.

- Prefer relational evidence such as:
  similar shapes,
  matching colors,
  repeated structures,
  containment,
  alignment,
  relative placement,
  symmetry,
  or differences between structurally similar regions.

- Distinguish observation from hypothesis.

- Reference object IDs explicitly in the evidence.

- Do not invent game mechanics that are unsupported by the image
  or structured observations.

- Return only the structure required by the supplied JSON schema.
"""

    response = client.chat.completions.create(
        model=LOCAL_MODEL,

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],

        temperature=0,

        max_tokens=2048,

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "scene_analysis",
                "schema": SCENE_SCHEMA,
            },
        },
    )

    raw = response.choices[0].message.content

    if not raw:
        raise RuntimeError(
            "Local VLM returned no content."
        )

    analysis = json.loads(raw)

    analysis = validate_scene_analysis(
        analysis,
        controlled_components,
        objects,
        traversable_color_evidence,
    )

    return analysis

class MyAgent2(Agent):
    MAX_ACTIONS = 20

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
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

    def is_done(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> bool:
        return latest_frame.state is GameState.WIN

    def infer_controlled_entity(self):
        if len(self.action_components) < 4:
            return

        component_sets = list(
            self.action_components.values()
        )

        controlled = set.intersection(
            *component_sets
        )

        self.controlled_component_ids = controlled

        print(
            "CONTROLLED ENTITY:",
            sorted(controlled)
        )

    def learn_traversable_colors(
        self,
        previous_frame,
        transition,
    ):
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
            movement.object_id
            for movement in movements
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

        objects = extract_objects(frame)

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
            else:
                self.track_controlled_entity(
                    transition,
                )

            self.learn_traversable_colors(
                self.previous_frame,
                transition,
            )

        self.previous_state = objects

        # Important: keep a copy because this is
        # the frame BEFORE the next action.
        self.previous_frame = frame.copy()

        return objects
    
    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:

        # Reset if needed
        if latest_frame.state in [
            GameState.NOT_PLAYED,
            GameState.GAME_OVER,
        ]:
            self.previous_state = None
            self.previous_action = None
            return GameAction.RESET

        # latest_frame.frame is a list of grids.
        # For now, examine the most recent one.
        if latest_frame.frame:
            frame = latest_frame.frame[-1]
            self.observe(frame)

        # once actions are exhausted, move on to analyze the scene
        if (
            self.mode == "DISCOVER"
            and len(self.action_vectors) >= 4
            and self.scene_analysis is None
        ):
            self.mode = "ANALYZE"

            self.scene_analysis = analyze_scene_vlm(
                frame=frame,
                objects=self.previous_state,
                action_vectors=self.action_vectors,
                controlled_components=(self.controlled_component_ids),
                traversable_color_evidence=(self.traversable_color_evidence),
            )

            print("\n=== VLM SCENE ANALYSIS ===")

            print(
                json.dumps(
                    self.scene_analysis,
                    indent=2,
                )
            )
            self.mode = "ACT"
            
        if self.mode == "ACT":

            goal = select_primary_goal(
                self.scene_analysis
            )

            if goal is not None:
                if goal["type"] in {
                    "reach_object",
                    "move_into_region",
                    "activate_object",
                }:
                    action = choose_goal_action(
                        frame=np.asarray(frame),
                        objects=self.previous_state,
                        controlled_ids=(
                            self.controlled_component_ids
                        ),
                        target_ids=goal["target_ids"],
                        action_vectors=self.action_vectors,
                        scene_analysis=self.scene_analysis,
                        traversable_color_evidence=(
                            self.traversable_color_evidence
                        ),
                    )

                    if action is not None:
                        self.previous_action = action
                        return action

            # No sufficiently confident goal.
            # Fall back to local exploration instead of returning None.
            action = choose_exploration_action(
                frame=np.asarray(frame),
                objects=self.previous_state,
                controlled_ids=(
                    self.controlled_component_ids
                ),
                action_vectors=self.action_vectors,
                scene_analysis=self.scene_analysis,
                traversable_color_evidence=(
                    self.traversable_color_evidence
                ),
            )

            if action is not None:
                self.previous_action = action
                return action

            # Absolute safety fallback:
            self.previous_action = GameAction.ACTION1
            return GameAction.ACTION1
        
        if self.mode == "DISCOVER":
            actions = [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
            ]

            action = actions[
                self.action_index % len(actions)
            ]

            self.action_index += 1

            self.previous_action = action

            return action
    
