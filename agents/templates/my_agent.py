import random
import time
from typing import Any
from collections import deque
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


class MyAgent(Agent):
    """Random agent — picks a random action each step.
    
    Modify this class to implement your own agent strategy!
    """

    MAX_ACTIONS = 100

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        # maintain actions done
        self.actions_tried = {}
        self.previous_key = None
        self.previous_action = None
        self.previous_diff = None
        self.transitions = {}
        self.available_by_state = {}
        self.previous_player_position = None
        self.spawn_position = None
        self.just_respawned = False
        self.shape_solution_source = None
        self.shape_solution_action = None
        self.mode = "EXPLORE"
        self.player_position_by_state = {}

    def is_done(self, frames, latest_frame):
        return latest_frame.state in [
            GameState.WIN,
            GameState.GAME_OVER,
        ]

    def frame_key(self, frame):
        final_grid = frame.frame[-1]
        
        game_grid = final_grid[:-5]

        return (
            frame.levels_completed,
            tuple(
                tuple(row)
                for row in game_grid
            ),
        )
    
    def path_to_frontier(self, start_key):
        queue = deque([start_key])

        # child_state -> parent_state
        parent = {start_key: None}

        # child_state -> action used to reach it
        via_action = {}

        while queue:
            state = queue.popleft()

            available = self.available_by_state.get(state, ())
            tried = self.actions_tried.get(state, set())

            # Does this state still have something unexplored?
            if any(action not in tried for action in available):
                path = []
                current = state

                while parent[current] is not None:
                    path.append(via_action[current])
                    current = parent[current]

                path.reverse()
                return path

            # Follow known transitions
            for (source, action), info in self.transitions.items():

                if source != state:
                    continue

                if not info["changed"]:
                    continue

                next_state = info["next_state"]

                if next_state == state:
                    continue

                if next_state not in parent:
                    parent[next_state] = state
                    via_action[next_state] = action
                    queue.append(next_state)

        return None
    def path_to_state(self, start_key, target_key):
        if start_key == target_key:
            return []

        queue = deque([start_key])

        parent = {
            start_key: None
        }

        via_action = {}

        while queue:
            state = queue.popleft()

            for (source, action), info in self.transitions.items():
                if source != state:
                    continue

                if not info["changed"]:
                    continue

                next_state = info["next_state"]

                if next_state in parent:
                    continue

                parent[next_state] = state
                via_action[next_state] = action

                if next_state == target_key:
                    path = []
                    current = next_state

                    while parent[current] is not None:
                        path.append(
                            via_action[current]
                        )
                        current = parent[current]

                    path.reverse()
                    return path

                queue.append(next_state)

        return None

    def find_components(self, grid, color):
        rows = len(grid)
        cols = len(grid[0])

        visited = set()
        components = []

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        for r in range(rows):
            for c in range(cols):

                if grid[r][c] != color:
                    continue

                if (r, c) in visited:
                    continue

                # Start a new component
                component = set()
                stack = [(r, c)]
                visited.add((r, c))

                while stack:
                    cr, cc = stack.pop()
                    component.add((cr, cc))

                    for dr, dc in directions:
                        nr = cr + dr
                        nc = cc + dc

                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and grid[nr][nc] == color
                            and (nr, nc) not in visited
                        ):
                            visited.add((nr, nc))
                            stack.append((nr, nc))

                components.append(component)

        return components
    
    def is_black_frame(self, grid, bbox, black=5):
        min_r, min_c, max_r, max_c = bbox

        border = []

        # top and bottom
        for c in range(min_c, max_c + 1):
            border.append(grid[min_r][c])
            border.append(grid[max_r][c])

        # left and right
        for r in range(min_r + 1, max_r):
            border.append(grid[r][min_c])
            border.append(grid[r][max_c])

        black_count = sum(
            pixel == black
            for pixel in border
        )

        return black_count / len(border) > 0.9
    
    def blue_inside_box(self, grid, bbox, blue=9):
        min_r, min_c, max_r, max_c = bbox

        pixels = set()

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):

                if grid[r][c] == blue:
                    pixels.add((r, c))

        return pixels
    
    def component_bbox(self, component):
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        return min_r, min_c, max_r, max_c

    def is_square_component(self, component):
        min_r, min_c, max_r, max_c = self.component_bbox(component)

        height = max_r - min_r + 1
        width = max_c - min_c + 1

        return height == width
    
    def normalize_shape(self, pixels):
        min_r = min(r for r, c in pixels)
        min_c = min(c for r, c in pixels)

        return {
            (r - min_r, c - min_c)
            for r, c in pixels
        }
    
    def print_shape(self, pixels):
        shape = self.normalize_shape(pixels)

        max_r = max(r for r, c in shape)
        max_c = max(c for r, c in shape)
        for r in range(max_r + 1):
            line = ""
            for c in range(max_c + 1):
                line += "#" if (r, c) in shape else "."
            print(line)

    def resize_shape(self, pixels):
        normalized = self.normalize_shape(pixels)

        resized = {
            (r // 2, c // 2)
            for r, c in normalized
        }

        return self.normalize_shape(resized)
    
    def shape_difference(self, shape_a, shape_b):
        return len(shape_a ^ shape_b)

    def bbox_center(self, bbox):
        min_r, min_c, max_r, max_c = bbox

        return (
            (min_r + max_r) // 2,
            (min_c + max_c) // 2,
        )
    
    def inside_bbox(self, component, bbox):
        min_r, min_c, max_r, max_c = bbox

        return all(
            min_r <= r <= max_r
            and min_c <= c <= max_c
            for r, c in component
        )
    def choose_goal_action(
        self,
        key,
        player_pos,
        goal_pos,
        available_actions
    ):
        pr, pc = player_pos
        gr, gc = goal_pos

        deltas = {
            GameAction.ACTION1: (-5, 0),
            GameAction.ACTION2: (5, 0),
            GameAction.ACTION3: (0, -5),
            GameAction.ACTION4: (0, 5),
        }

        candidates = []

        for action in available_actions:

            info = self.transitions.get(
                (key, action)
            )

            # Known wall / impossible move
            if (
                info is not None
                and not info["changed"]
            ):
                continue

            # If we already know the resulting state,
            # use its real player position.
            if info is not None:
                next_key = info["next_state"]

                next_pos = (
                    self.player_position_by_state.get(
                        next_key
                    )
                )
            else:
                next_pos = None

            if next_pos is not None:
                nr, nc = next_pos
                knowledge_penalty = 0

            else:
                dr, dc = deltas[action]

                nr = pr + dr
                nc = pc + dc

                # Prefer known transitions over guesses
                # when distance is equal.
                knowledge_penalty = 1

            distance = (
                abs(nr - gr)
                + abs(nc - gc)
            )

            candidates.append(
                (
                    distance,
                    knowledge_penalty,
                    action,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (x[0], x[1])
        )

        return candidates[0][2]

    def append_frame(self, frame: FrameData) -> None:
        # Let the base Agent store/record the frame normally
        super().append_frame(frame)

        # If an actual action was previously taken,
        # record its result.
        if (
            self.previous_key is not None
            and self.previous_action is not None
        ):
            new_key = self.frame_key(frame)

            changed = new_key != self.previous_key

            transition = (
                self.previous_key,
                self.previous_action,
            )

            self.transitions[transition] = {
                "next_state": new_key,
                "changed": changed,
                "result": frame.state,
            }

            print(
                f"RESULT: {self.previous_action.name}"
                f" -> changed={changed}"
                f", state={frame.state.name}"
                f", next_state={hash(new_key)}"
            )

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            # identify square components
            grid = latest_frame.frame[-1]
            black_components = self.find_components(grid,5)

            square_components = [
                comp
                for comp in black_components
                if self.is_square_component(comp)
            ]

            square_bboxes = [
                self.component_bbox(comp)
                for comp in square_components
            ]

            print(f"Found {len(square_components)} square black regions")

            # identify player
            blue_components = self.find_components(grid, 9)

            world_components = [
                comp for comp in blue_components
                if not any(
                    self.inside_bbox(comp, bbox)
                    for bbox in square_bboxes
                )
            ]

            player = world_components[0]
            player_bbox = self.component_bbox(player)
            player_position = self.bbox_center(player_bbox)
            if self.spawn_position is None:
                self.spawn_position = player_position

            self.just_respawned = False

            if self.previous_player_position is not None:
                old_r, old_c = self.previous_player_position
                new_r, new_c = player_position

                dr = abs(new_r - old_r)
                dc = abs(new_c - old_c)

                # Normal movement is 0 or 5 cells.
                # A larger jump is probably a respawn/teleport.
                if dr > 5 or dc > 5:
                    self.just_respawned = True
                    bad_transition = (self.previous_key,self.previous_action)
                    if bad_transition in self.transitions:
                        del self.transitions[bad_transition]
                    print(
                        "RESPAWN DETECTED:",
                        self.previous_player_position,
                        "->",
                        player_position
                    )

            self.previous_player_position = player_position

            if self.just_respawned:
                if self.shape_solution_source is not None:
                    self.mode = "RECOVER_SHAPE"
                else:
                    self.mode = "EXPLORE"

            print(f"World blue components: {len(world_components)}")
            for i, comp in enumerate(world_components):
                print(
                    f"World component {i}: "
                    f"size={len(comp)}, "
                    f"bbox={self.component_bbox(comp)}"
                )

            # find goal and target shapes by finding black squares
            for i, comp in enumerate(square_components):
                bbox = self.component_bbox(comp)
                blue_pixels = self.blue_inside_box(grid, bbox)
                if len(blue_pixels) > 6:
                    current_shape = self.resize_shape(blue_pixels)
                else:
                  goal_shape = self.normalize_shape(blue_pixels)
                  door_position = self.bbox_center(bbox)
            diff = self.shape_difference(goal_shape, current_shape)
            print(f"Diff: {diff}")

            if (
                self.previous_diff is not None
                and diff < self.previous_diff):
                print("SHAPE IMPROVED:",self.previous_diff,"->",diff)
                if diff == 0:
                    self.shape_solution_source = self.previous_key
                    self.shape_solution_action = self.previous_action

                    print(
                        "SAVED SHAPE SOLUTION:",
                        self.shape_solution_action
                    )

            self.previous_diff = diff
            # check available actions
            key = self.frame_key(latest_frame)
            available_actions = [a if isinstance(a, GameAction) else GameAction.from_id(a)
                                for a in latest_frame.available_actions]
            available_actions = [a for a in available_actions if a is not GameAction.RESET]

            self.available_by_state[key] = tuple(available_actions)
            tried = self.actions_tried.setdefault(key, set())
            untried = [action for action in available_actions
                      if action not in tried]
            self.player_position_by_state[key] = (
                player_position
            )
            action = None
            if diff == 0:
                print("SHAPE MATCHED")
                print("Player:", player_position)
                print("Goal:", door_position)
                self.mode = "GOAL"
                action = self.choose_goal_action(
                    key,
                    player_position,
                    door_position,
                    available_actions
                )
            else:
                if self.mode == "RECOVER_SHAPE":
                    if key == self.shape_solution_source:
                        action = self.shape_solution_action
                        print("REPLAYING SHAPE ACTION:", action.name)
                    else:
                        path = self.path_to_state(key,self.shape_solution_source)
                        if path:
                            action = path[0]
                            print("RETURNING TO SHAPE BUTTON:", action.name)
                        else:
                            print(
                                "Could not find route to known "
                                "shape button - returning to exploration"
                            )
                            self.mode = "EXPLORE"
                if action is None:
                    self.mode = "EXPLORE"
                    # explore with BFS
                    if not available_actions:
                        return GameAction.RESET
                    if untried:
                        action = random.choice(untried)
                    else:
                        path = self.path_to_frontier(key)
                        if path:
                            action = path[0]
                            print(
                                f"\nBACKTRACKING:"
                                f" {action.name}"
                                f" toward nearest frontier"
                            )
                        else:
                            print("\nNO KNOWN FRONTIER")
                            action = random.choice(available_actions)

            print(
                f"\nMODE: {self.mode}"
                f"\nSTATE: {hash(key)}"
                f"\nAvailable: {[a.name for a in available_actions]}"
                f"\nAlready tried: {[a.name for a in tried]}"
                f"\nChosen: {action.name}"
            )
            # common bookkeeping
            self.previous_key = key
            self.previous_action = action
            tried.add(action)
        if action.is_simple():
            action.reasoning = (f"Exploring action {action.value}")
        elif action.is_complex():
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "RNG said so!",
            }
        return action