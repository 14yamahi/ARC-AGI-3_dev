"""Pure tracking regressions, runnable without Kaggle or a model endpoint.

Load the solver's definitions without its import-time HTTP client and optional
Kaggle imports. The production implementations themselves are exercised.
"""
import ast
import asyncio
import contextlib
import __future__
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import traceback
from unittest.mock import patch

import numpy as np


class Action(Enum):
    # Match ARCEngine: tuple construction with _value_ set during __init__.
    # An IntEnum would incorrectly allow Action(0), masking the Kaggle failure.
    RESET = (0, object)
    ACTION1 = (1, object)
    ACTION2 = (2, object)
    ACTION3 = (3, object)
    ACTION4 = (4, object)

    def __init__(self, action_id, action_type):
        self._value_ = action_id
        self.action_type = action_type

    @classmethod
    def from_id(cls, action_id):
        for action in cls:
            if action.value == action_id:
                return action
        raise ValueError(f'No GameAction with id {action_id}')


def load_solver():
    path = Path(__file__).resolve().parents[2] / 'agents/templates/my_agent2.py'
    tree = ast.parse(path.read_text())
    names = {
        'GameObject', 'Movement', 'Transition', 'ObjectTracker', 'MyAgentCore',
        'object_hash', 'normalized_shape', 'translate_pixels', 'maximum_assignment',
        'match_objects', 'describe_transition', 'extract_objects', 'group_movements',
        'resolve_goal_target_ids', 'capture_goal_target_tracks', 'get_object_pixels',
        'get_controlled_pixels', 'pixel_distance', 'destination_colors',
        'bbox_gap', 'expand_goal_target_region',
        'select_primary_goal', 'MyAgentSolver',
        'validate_scene_analysis', 'SceneAnalysisValidationError',
        'detect_ui_candidates',
        'vlm_frame_id_view',
        'choose_goal_action_bfs', 'choose_frontier_action', 'choose_exploration_action',
        'plan_to_nearest_frontier', 'get_frontier_actions', 'inverse_action',
    }
    namespace = dict(dataclass=dataclass, np=np, hashlib=hashlib,
                     deque=deque, defaultdict=defaultdict,
                     arcengine=SimpleNamespace(GameAction=Action, GameState=SimpleNamespace(GAME_OVER='over', WIN='win'),
                                               ActionInput=lambda **kw: SimpleNamespace(**kw)),
                     taaf=SimpleNamespace(game=SimpleNamespace(Game=object)), Solver=object,
                     asyncio=asyncio, json=json, contextlib=contextlib, traceback=traceback)
    namespace['normalize_scene_analysis'] = lambda analysis: analysis
    selected = ast.Module(
        body=[n for n in tree.body if getattr(n, 'name', None) in names],
        type_ignores=[],
    )
    # The local test environment may be Python 3.8. Compile the extracted
    # definitions with PEP 563 semantics so modern annotations such as
    # set[tuple[int, int]] are never evaluated at runtime.
    code = compile(
        selected,
        str(path),
        'exec',
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    return namespace


M = load_solver()


def obj(pixels, color=1, frame_id=0):
    pixels = set(pixels)
    ys, xs = zip(*pixels)
    return M['GameObject'](frame_id, color, pixels,
                          (min(ys), min(xs), max(ys), max(xs)),
                          (sum(ys) / len(ys), sum(xs) / len(xs)),
                          M['object_hash'](pixels, color), [], [])


class TestPersistentTracking(unittest.TestCase):
    def run_discovery(self, available, deltas):
        core = M['MyAgentCore']()
        position = (8, 8)
        analysis = {
            'wall_candidates': [], 'ui_candidates': [], 'important_objects': [],
            'goal_scores': {'unknown': {'confidence': 1, 'target_ids': []}},
        }
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(
            M, analyze_scene_vlm=lambda **kwargs: analysis,
        ):
            for _ in range(core.MAX_DISCOVERY_ACTIONS + 1):
                frame = np.zeros((20, 20), dtype=int)
                y, x = position
                frame[y:y+2, x:x+2] = 1
                action = core.choose_action(frame, available)
                self.assertIn(action.value, available)
                if core.mode != 'DISCOVER':
                    break
                dx, dy = deltas.get(action, (0, 0))
                position = (max(1, min(17, y + dy)), max(1, min(17, x + dx)))
        return core

    def test_discovery_finishes_with_one_or_two_controls(self):
        for available, deltas in (
            ([4], {Action.ACTION4: (2, 0)}),
            ([3, 4], {Action.ACTION3: (-2, 0), Action.ACTION4: (2, 0)}),
        ):
            with self.subTest(available=available):
                core = self.run_discovery(available, deltas)
                self.assertEqual(core.mode, 'ACT')
                self.assertLess(core.discovery_steps, core.MAX_DISCOVERY_ACTIONS)
                self.assertEqual(len(core.controlled_component_ids), 1)
                self.assertEqual(set(core.action_vectors), set(deltas))

    def test_discovery_budget_accepts_blocked_directions(self):
        core = self.run_discovery(
            [1, 2, 3, 4], {Action.ACTION3: (-2, 0), Action.ACTION4: (2, 0)},
        )
        self.assertEqual(core.mode, 'ACT')
        self.assertEqual(core.discovery_steps, core.MAX_DISCOVERY_ACTIONS)
        self.assertEqual(len(core.controlled_component_ids), 1)
        self.assertEqual(set(core.action_vectors), {Action.ACTION3, Action.ACTION4})

    def test_discovery_budget_finishes_without_movement_evidence(self):
        core = self.run_discovery([1, 2, 3, 4], {})
        self.assertEqual(core.mode, 'ACT')
        self.assertEqual(core.discovery_steps, core.MAX_DISCOVERY_ACTIONS)
        self.assertFalse(core.controlled_component_ids)
        self.assertFalse(core.action_vectors)
        self.assertEqual(set(core.discovery_attempts), set(list(Action)[1:]))

    def test_discovery_prefers_continuation_over_learned_reverse(self):
        core = M['MyAgentCore']()
        frame = np.zeros((12, 12), dtype=int)
        frame[4:6, 4:6] = 1
        core.action_vectors = {Action.ACTION1: (0, -2), Action.ACTION2: (0, 2)}
        core.previous_action = Action.ACTION1
        # Leave other available controls unknown so discovery remains active.
        action = core.choose_action(frame, [1, 2, 3, 4])
        self.assertEqual(action, Action.ACTION1)

    def test_discovery_budget_applies_to_unresolved_controlled_track(self):
        core = M['MyAgentCore']()
        core.controlled_track_ids = {'missing'}
        core.discovery_steps = core.MAX_DISCOVERY_ACTIONS
        analysis = {'wall_candidates': [], 'ui_candidates': [],
                    'goal_scores': {'unknown': {'confidence': 1, 'target_ids': []}}}
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(
            M, analyze_scene_vlm=lambda **kwargs: analysis,
        ):
            core.choose_action(np.zeros((4, 4), dtype=int), [1])
        self.assertEqual(core.mode, 'ACT')
        self.assertFalse(core.controlled_track_ids)

    def test_numeric_action_ids_use_engine_lookup(self):
        with self.assertRaises(ValueError):
            Action(0)
        core = M['MyAgentCore']()
        frame = np.zeros((4, 4), dtype=int)
        self.assertEqual(core.choose_action(frame, [0, 1, 2, 3, 4]), Action.ACTION1)

    def test_reset_only_availability(self):
        core = M['MyAgentCore']()
        self.assertEqual(core.choose_action(np.zeros((4, 4), dtype=int), [0]), Action.RESET)

    def test_enum_and_mixed_availability(self):
        for available in ([Action.ACTION3], [0, Action.ACTION3]):
            core = M['MyAgentCore']()
            self.assertEqual(core.choose_action(np.zeros((4, 4), dtype=int), available), Action.ACTION3)

    def test_invalid_and_empty_availability_fail_explicitly(self):
        for available in ([99], []):
            with self.assertRaises(ValueError):
                M['MyAgentCore']().choose_action(np.zeros((4, 4), dtype=int), available)

    def test_global_assignment_and_unmatched(self):
        self.assertEqual(set(M['maximum_assignment']([[.9, .8], [.85, .1]])), {(0, 1), (1, 0)})
        self.assertEqual(M['maximum_assignment']([[-1], [-1]]), [])
        self.assertEqual(M['maximum_assignment']([[]]), [])

    def test_identical_objects_keep_distinct_ids_after_frame_renumbering(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(0, 0)}, frame_id=1), obj({(0, 10)}, frame_id=2)])
        after = tracker.update([obj({(0, 10)}, frame_id=99), obj({(0, 0)}, frame_id=3)])
        self.assertEqual(after[0].track_id, before[1].track_id)
        self.assertEqual(after[1].track_id, before[0].track_id)
        self.assertNotEqual(after[0].track_id, after[1].track_id)
        self.assertEqual(M['resolve_goal_target_ids'](after, [before[0].track_id]), [3])

    def test_recolor_and_move_preserve_identity_and_both_events(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(0, 0), (0, 1)})])
        after = tracker.update([obj({(0, 5), (0, 6)}, color=2)], [before[0].track_id], (5, 0))
        self.assertEqual(before[0].track_id, after[0].track_id)
        transition = M['describe_transition'](before, Action.ACTION4, after)
        self.assertEqual(len(transition.changed_objects), 1)
        self.assertEqual(transition.moved_objects[0].delta, (5, 0))

    def test_deformation_preserves_identity(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(0, 0), (0, 1), (1, 0), (1, 1)})])
        after = tracker.update([obj({(0, 0), (0, 1), (1, 0)})])
        self.assertEqual(before[0].track_id, after[0].track_id)
        self.assertEqual(len(M['describe_transition'](before, None, after).changed_objects), 1)

    def test_symmetric_ambiguity_is_unresolved(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(0, 0)}), obj({(0, 4)}, frame_id=1)])
        after = tracker.update([obj({(0, 2)})])
        self.assertIsNone(after[0].track_id)
        self.assertEqual(after[0].tracking_status, 'ambiguous')
        self.assertEqual(M['resolve_goal_target_ids'](after, [before[0].track_id]), [])

    def test_temporary_missing_and_return(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(2, 2)})])
        tracker.update([])
        self.assertEqual(tracker.events[0]['type'], 'unresolved')
        after = tracker.update([obj({(2, 2)}, frame_id=5)])
        self.assertEqual(before[0].track_id, after[0].track_id)

    def test_prolonged_ambiguity_does_not_mint_replacement_identity(self):
        tracker = M['ObjectTracker']()
        tracker.update([obj({(0, 0)}), obj({(0, 4)}, frame_id=1)])
        for _ in range(8):
            after = tracker.update([obj({(0, 2)})])
            self.assertIsNone(after[0].track_id)

    def test_ambiguous_tracks_can_be_disambiguated(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(0, 0)}), obj({(0, 4)}, frame_id=1)])
        tracker.update([obj({(0, 2)})])
        after = tracker.update([obj({(0, 0)}), obj({(0, 4)}, frame_id=1)])
        self.assertEqual([o.track_id for o in after], [o.track_id for o in before])

    def test_expired_track_is_not_reused(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(2, 2)})])
        for _ in range(4):
            tracker.update([])
        after = tracker.update([obj({(2, 2)})])
        self.assertNotEqual(before[0].track_id, after[0].track_id)

    def test_split_and_merge_have_fresh_ids_and_lineage(self):
        tracker = M['ObjectTracker']()
        before = tracker.update([obj({(0, 0), (0, 1), (0, 2), (0, 3)})])
        split = tracker.update([obj({(0, 0), (0, 1)}), obj({(0, 2), (0, 3)}, color=2, frame_id=1)])
        for child in split:
            self.assertEqual(child.parent_track_ids, (before[0].track_id,))
            self.assertNotEqual(child.track_id, before[0].track_id)
        merged = tracker.update([obj({(0, 0), (0, 1), (0, 2), (0, 3)})])
        self.assertEqual(set(merged[0].parent_track_ids), {child.track_id for child in split})
        self.assertNotIn(merged[0].track_id, {child.track_id for child in split})

    def test_background_not_split_by_player_motion(self):
        tracker = M['ObjectTracker']()
        frame = np.zeros((12, 12), dtype=int)
        frame[4:6, 4:6] = 1
        before = tracker.update(M['extract_objects'](frame))
        player = next(o for o in before if o.color == 1)
        background = next(o for o in before if o.color == 0)
        frame[4:6, 4:6] = 0
        frame[4:6, 6:8] = 1
        after = tracker.update(M['extract_objects'](frame), [player.track_id], (2, 0))
        self.assertEqual(next(o for o in after if o.color == 0).track_id, background.track_id)
        self.assertEqual(next(o for o in after if o.color == 1).track_id, player.track_id)
        self.assertFalse(any(event['type'] == 'split_or_merge' for event in tracker.events))

    def test_capture_and_partial_target_resolution(self):
        tracker = M['ObjectTracker']()
        objects = tracker.update([obj({(0, 0)}, frame_id=10), obj({(0, 10)}, frame_id=11)])
        scene = {'goal_scores': {'reach_object': {'target_ids': [10]}}}
        tracks = M['capture_goal_target_tracks'](scene, objects)['reach_object']
        self.assertEqual(tracks, [objects[0].track_id])
        self.assertEqual(M['resolve_goal_target_ids'](objects[:1], [o.track_id for o in objects]), [])

    def test_wall_and_ui_roles_follow_renumbering_through_observe(self):
        core = M['MyAgentCore']()
        frame = np.zeros((12, 12), dtype=int)
        frame[1, 1] = 6  # Removing this earlier component renumbers later IDs.
        frame[4, 4] = 5
        frame[8, 8] = 11
        objects = core.observe(frame)
        wall = next(o for o in objects if o.color == 5)
        ui = next(o for o in objects if o.color == 11)
        analysis = {
            'wall_candidates': [wall.id], 'ui_candidates': [ui.id],
            'important_objects': [],
            'goal_scores': {'unknown': {'confidence': 1, 'target_ids': []}},
        }
        core.mode = 'ANALYZE'
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(
            M, analyze_scene_vlm=lambda **kwargs: analysis,
        ):
            core.choose_action(frame, [1])
            frame[1, 1] = 0
            after = core.observe(frame)
        new_wall = next(o for o in after if o.track_id == wall.track_id)
        new_ui = next(o for o in after if o.track_id == ui.track_id)
        self.assertNotEqual(new_wall.id, wall.id)
        self.assertNotEqual(new_ui.id, ui.id)
        self.assertEqual(core.scene_analysis['wall_candidates'], [new_wall.id])
        self.assertEqual(core.scene_analysis['ui_candidates'], [new_ui.id])
        core.current_goal = {'type': 'activate_object', 'target_tracks': [wall.track_id]}
        pending = core.begin_interaction_observation([new_wall.id], Action.ACTION1)
        self.assertEqual(pending['ui_tracks'], {ui.track_id})

    def test_missing_role_does_not_transfer_to_reused_id_or_drop_other_walls(self):
        core = M['MyAgentCore']()
        wall, other_wall, ui = [obj({(0, x)}, frame_id=i)
                                for i, x in enumerate((0, 10, 20))]
        wall.track_id, other_wall.track_id, ui.track_id = 'wall', 'other', 'ui'
        core.scene_analysis = {'wall_candidates': [0, 1], 'ui_candidates': [2]}
        core.capture_scene_role_tracks([wall, other_wall, ui])
        replacement = obj({(1, 1)}, frame_id=0)
        replacement.track_id = 'replacement'
        other_wall.id = 7
        core.resolve_scene_role_ids([replacement, other_wall])
        self.assertEqual(core.scene_analysis['wall_candidates'], [7])
        self.assertEqual(core.scene_analysis['ui_candidates'], [])
        wall.id, ui.id = 8, 9
        core.resolve_scene_role_ids([replacement, wall, other_wall, ui])
        self.assertEqual(core.scene_analysis['wall_candidates'], [7, 8])
        self.assertEqual(core.scene_analysis['ui_candidates'], [9])

    def test_reanalysis_replaces_roles_and_rejects_ambiguous_or_missing_ids(self):
        core = M['MyAgentCore']()
        wall = obj({(0, 0)}, frame_id=1)
        wall.track_id = 'wall'
        ambiguous = obj({(0, 4)}, frame_id=2)
        core.scene_analysis = {'wall_candidates': [1, 2, 999], 'ui_candidates': []}
        core.capture_scene_role_tracks([wall, ambiguous])
        self.assertEqual(core.scene_analysis['wall_candidates'], [1])
        core.scene_analysis = {'wall_candidates': [], 'ui_candidates': [1]}
        core.capture_scene_role_tracks([wall, ambiguous])
        wall.id = 5
        core.resolve_scene_role_ids([wall, ambiguous])
        self.assertEqual(core.scene_analysis['wall_candidates'], [])
        self.assertEqual(core.scene_analysis['ui_candidates'], [5])
        self.assertFalse(M['MyAgentCore'](1).scene_role_tracks)

    def test_role_does_not_automatically_transfer_to_lineage_replacement(self):
        core = M['MyAgentCore']()
        parent = obj({(0, 0), (0, 1)})
        parent.track_id = 'parent'
        core.scene_analysis = {'wall_candidates': [parent.id], 'ui_candidates': []}
        core.capture_scene_role_tracks([parent])
        child = obj({(0, 0)})
        child.track_id = 'child'
        child.parent_track_ids = ('parent',)
        child.tracking_status = 'lineage'
        core.resolve_scene_role_ids([child])
        self.assertEqual(core.scene_analysis['wall_candidates'], [])

    def test_target_region_expands_adjacent_marker_components_for_bfs(self):
        frame = np.full((12, 8), 3, dtype=int)
        # A 5x5 controlled object moves upward by five pixels. Its final
        # footprint contacts the white marker and two light-gray pixels.
        frame[6:11, 0:5] = 8
        frame[2, 2] = 0
        frame[2, 1] = 1
        frame[3, 2] = 1
        # A directly adjacent black obstacle must not join the target region.
        frame[0, 3] = 5
        objects = M['ObjectTracker']().update(M['extract_objects'](frame))
        player = next(o for o in objects if o.color == 8)
        target = next(o for o in objects if o.color == 0)
        companions = [o for o in objects if o.color == 1]
        obstacle = next(o for o in objects if o.color == 5)

        without_region = M['choose_goal_action_bfs'](
            frame, objects, [player.id], [target.id],
            {Action.ACTION1: (0, -5)}, {3: 2}, set(),
        )
        region = M['expand_goal_target_region'](
            objects, [target.id], [player.id], [], [obstacle.id], {3: 2},
        )
        with_region = M['choose_goal_action_bfs'](
            frame, objects, [player.id], region,
            {Action.ACTION1: (0, -5)}, {3: 2}, set(),
        )

        self.assertIsNone(without_region)
        self.assertEqual(region, sorted([target.id, *(o.id for o in companions)]))
        self.assertNotIn(obstacle.id, region)
        self.assertEqual(with_region, Action.ACTION1)

    def test_target_region_does_not_absorb_floor_or_large_neighbor(self):
        target = obj({(3, 3)}, color=0, frame_id=1)
        floor = obj({(3, 4)}, color=3, frame_id=2)
        large = obj({(4, col) for col in range(20)}, color=1, frame_id=3)
        target.track_id = 'target'
        floor.track_id = 'floor'
        large.track_id = 'large'
        region = M['expand_goal_target_region'](
            [target, floor, large], [target.id], [], [], [], {3: 2},
        )
        self.assertEqual(region, [target.id])

    def test_validation_rejects_ambiguous_target(self):
        objects = [obj({(0, 0)}, frame_id=5)]
        analysis = {'wall_candidates': [], 'ui_candidates': [], 'important_objects': [5],
                    'goal_scores': {'reach_object': {'target_ids': [5], 'confidence': .9, 'evidence': ''}}}
        result = M['validate_scene_analysis'](analysis, set(), objects, {})
        self.assertEqual(result['goal_scores']['reach_object']['target_ids'], [])
        self.assertEqual(result['goal_scores']['reach_object']['confidence'], 0)

    def test_validation_retries_when_goal_targets_controlled_or_ui_components(self):
        frame = np.zeros((12, 12), dtype=int)
        player = obj({(5, 5)}, frame_id=1)
        declared_ui = obj({(6, 6)}, frame_id=2)
        edge_ui = obj({(10, 4)}, frame_id=3)
        target = obj({(4, 4)}, frame_id=4)
        for item, track in zip(
            (player, declared_ui, edge_ui, target),
            ('player', 'declared-ui', 'edge-ui', 'target'),
        ):
            item.track_id = track
        analysis = {
            'wall_candidates': [], 'ui_candidates': [declared_ui.id],
            'important_objects': [declared_ui.id, edge_ui.id, target.id],
            'goal_scores': {
                'reach_object': {
                    'target_ids': [player.id, declared_ui.id, edge_ui.id, target.id],
                    'confidence': .9, 'evidence': '',
                },
            },
        }

        with self.assertRaises(M['SceneAnalysisValidationError']) as raised:
            M['validate_scene_analysis'](
                analysis, {player.id}, [player, declared_ui, edge_ui, target], {}, frame,
            )

        feedback = raised.exception.feedback
        rejected = feedback['rejected_target_ids']['reach_object']
        self.assertIn('controlled_component', rejected[player.id])
        self.assertIn('declared_ui_candidate', rejected[declared_ui.id])
        self.assertIn('edge_ui_candidate', rejected[edge_ui.id])
        self.assertEqual(feedback['deterministic_edge_ui_ids'], [edge_ui.id])

    def test_validation_preserves_valid_goal_and_promotes_edge_components_to_ui(self):
        frame = np.zeros((12, 12), dtype=int)
        edge_ui = obj({(0, 3)}, frame_id=1)
        target = obj({(5, 5)}, frame_id=2)
        edge_ui.track_id, target.track_id = 'edge-ui', 'target'
        analysis = {
            'wall_candidates': [], 'ui_candidates': [], 'important_objects': [edge_ui.id, target.id],
            'goal_scores': {
                'reach_object': {'target_ids': [target.id], 'confidence': .9, 'evidence': ''},
            },
        }
        result = M['validate_scene_analysis'](
            analysis, set(), [edge_ui, target], {}, frame,
        )
        self.assertEqual(result['ui_candidates'], [edge_ui.id])
        self.assertEqual(result['important_objects'], [target.id])
        self.assertEqual(result['goal_scores']['reach_object']['target_ids'], [target.id])

    def test_validation_rejects_model_declared_obstacle_goal(self):
        frame = np.full((12, 12), 4, dtype=int)
        player = obj({(6, 6)}, color=1, frame_id=1)
        wall = obj({(5, 5), (5, 6)}, color=5, frame_id=2)
        target = obj({(4, 4)}, color=9, frame_id=3)
        for item, track in zip((player, wall, target), ('player', 'wall', 'target')):
            item.track_id = track
        analysis = {
            'wall_candidates': [], 'ui_candidates': [],
            'important_objects': [wall.id, target.id],
            'object_roles': [
                {'object_id': wall.id, 'role': 'obstacle', 'confidence': .9, 'evidence': 'blocks route'},
            ],
            'goal_scores': {
                'reach_object': {'target_ids': [wall.id, target.id], 'confidence': .9, 'evidence': ''},
            },
        }

        with self.assertRaises(M['SceneAnalysisValidationError']) as raised:
            M['validate_scene_analysis'](analysis, {player.id}, [player, wall, target], {}, frame)

        self.assertIn(
            'declared_wall_or_obstacle',
            raised.exception.feedback['rejected_target_ids']['reach_object'][wall.id],
        )

    def test_validation_rejects_separate_background_colour_component_goal(self):
        frame = np.full((12, 12), 4, dtype=int)
        player = obj({(6, 6)}, color=1, frame_id=1)
        background_wall = obj({(5, 5)}, color=4, frame_id=2)
        target = obj({(4, 4)}, color=9, frame_id=3)
        for item, track in zip((player, background_wall, target), ('player', 'background-wall', 'target')):
            item.track_id = track
        analysis = {
            'wall_candidates': [], 'ui_candidates': [],
            'important_objects': [background_wall.id, target.id],
            'goal_scores': {
                'reach_object': {
                    'target_ids': [background_wall.id, target.id], 'confidence': .9, 'evidence': '',
                },
            },
        }

        with self.assertRaises(M['SceneAnalysisValidationError']) as raised:
            M['validate_scene_analysis'](
                analysis, {player.id}, [player, background_wall, target], {}, frame,
            )

        self.assertIn(
            'background_color_component',
            raised.exception.feedback['rejected_target_ids']['reach_object'][background_wall.id],
        )

    def test_vlm_context_replaces_tracking_ids_with_current_frame_ids(self):
        context = {
            'frame_id': 7,
            'track_id': 'current',
            'target_tracks': ['current', 'missing'],
            'relation': {'display_track': 'current', 'gameplay_track': 'missing'},
            'parent_track_ids': ['old'],
            'tracking_status': 'matched',
        }
        result = M['vlm_frame_id_view'](context, {'current': 7})
        self.assertEqual(result['frame_id'], 7)
        self.assertEqual(result['target_frame_ids'], [7])
        self.assertEqual(result['target_not_visible_count'], 1)
        self.assertEqual(result['relation']['display_frame_id'], 7)
        self.assertNotIn('gameplay_frame_id', result['relation'])
        self.assertNotIn('track_id', result)
        self.assertNotIn('parent_track_ids', result)

    def test_marker_uses_overlap_fallback_when_coarse_moves_skip_its_boundary(self):
        frame = np.full((64, 64), 4, dtype=int)
        frame[8:50, 14:54] = 3
        # The central obstruction leaves a route around it, but a 5-pixel
        # player can only end two pixels away from this compact cross or on it.
        frame[30:45, 24:34] = 4
        orange = obj(
            {(y, x) for y in range(45, 47) for x in range(34, 39)},
            color=2, frame_id=10,
        )
        blue = obj(
            {(y, x) for y in range(47, 50) for x in range(34, 39)},
            color=9, frame_id=11,
        )
        white = obj({(31, 21), (32, 21), (32, 22)}, color=0, frame_id=7)
        gray_a = obj({(32, 20)}, color=1, frame_id=8)
        gray_b = obj({(33, 21)}, color=1, frame_id=9)
        objects = [orange, blue, white, gray_a, gray_b]
        for item, track in zip(objects, ('orange', 'blue', 'white', 'gray-a', 'gray-b')):
            item.track_id = track
        core = M['MyAgentCore']()
        core.controlled_component_ids = {orange.id, blue.id}
        core.scene_analysis = {
            'wall_candidates': [], 'ui_candidates': [],
            'object_roles': [
                {'object_id': item.id, 'role': 'marker', 'confidence': .8, 'evidence': 'cross'}
                for item in (white, gray_a, gray_b)
            ],
        }
        actions = {
            Action.ACTION1: (0, -5), Action.ACTION2: (0, 5),
            Action.ACTION3: (-5, 0), Action.ACTION4: (5, 0),
        }
        target_ids = [white.id, gray_a.id, gray_b.id]

        self.assertFalse(core.goal_allows_overlap('reach_object', target_ids))
        self.assertTrue(core.can_try_destination_overlap('reach_object', target_ids, frame, objects))
        self.assertIsNone(M['choose_goal_action_bfs'](
            frame, objects, core.controlled_component_ids, target_ids,
            actions, {3: 4}, set(), allow_target_overlap=False,
        ))
        self.assertIsNotNone(M['choose_goal_action_bfs'](
            frame, objects, core.controlled_component_ids, target_ids,
            actions, {3: 4}, set(), allow_target_overlap=True,
        ))

    def test_overlap_fallback_rejects_declared_obstacles(self):
        frame = np.full((12, 12), 4, dtype=int)
        player = obj({(6, 6)}, color=1, frame_id=1)
        target = obj({(5, 5)}, color=9, frame_id=2)
        core = M['MyAgentCore']()
        core.controlled_component_ids = {player.id}
        core.scene_analysis = {
            'wall_candidates': [], 'ui_candidates': [],
            'object_roles': [
                {'object_id': target.id, 'role': 'obstacle', 'confidence': .9, 'evidence': 'wall'},
            ],
        }
        self.assertFalse(core.can_try_destination_overlap(
            'reach_object', [target.id], frame, [player, target],
        ))

    def test_inset_widget_inside_edge_panel_is_protected_ui_not_goal(self):
        frame = np.zeros((64, 64), dtype=int)
        player = obj({(32, 32)}, color=1, frame_id=1)
        panel_pixels = {(y, x) for y in range(53, 63) for x in range(1, 11)}
        panel = obj(panel_pixels, color=5, frame_id=2)
        # This status pattern is three cells from the outer edge, as in LS20,
        # so an edge-margin-only detector would wrongly make it a destination.
        status = obj({(55, 3), (56, 3), (56, 4)}, color=9, frame_id=3)
        world_goal = obj({(20, 20)}, color=9, frame_id=4)
        analysis = {
            'wall_candidates': [], 'ui_candidates': [],
            'important_objects': [status.id, world_goal.id],
            'goal_scores': {
                'reach_object': {
                    'confidence': .9, 'target_ids': [status.id], 'evidence': 'blue',
                },
            },
        }
        with self.assertRaises(M['SceneAnalysisValidationError']) as raised:
            M['validate_scene_analysis'](
                analysis, {player.id}, [player, panel, status, world_goal], {}, frame,
            )
        feedback = raised.exception.feedback
        self.assertIn(status.id, feedback['deterministic_edge_ui_ids'])
        self.assertIn('edge_ui_candidate', feedback['rejected_target_ids']['reach_object'][status.id])

    def test_state_display_is_excluded_from_post_interaction_candidates(self):
        core = M['MyAgentCore']()
        frame = np.zeros((16, 16), dtype=int)
        display = obj({(4, 3)}, color=9, frame_id=1)
        display.track_id = 'display'
        gameplay = obj({(8, 8)}, color=9, frame_id=2)
        gameplay.track_id = 'goal'
        core.state_display_tracks = {'display'}
        candidates = core.derive_post_interaction_candidates(
            [display, gameplay],
            {'target_tracks': (), 'ui_tracks': set(), 'snapshots': {}},
            [
                {'track_id': 'display', 'kind': 'changed', 'area': 'gameplay'},
                {'track_id': 'goal', 'kind': 'changed', 'area': 'gameplay'},
            ], frame,
        )
        self.assertEqual([item['track_id'] for item in candidates], ['goal'])

    def test_screen_anchored_object_requires_repeated_player_motion_before_ui_promotion(self):
        core = M['MyAgentCore']()
        frame = np.zeros((20, 20), dtype=int)
        player = obj({(10, 10)}, color=1, frame_id=1)
        widget = obj({(4, 4), (4, 5)}, color=9, frame_id=2)
        player.track_id, widget.track_id = 'player', 'widget'
        core.controlled_track_ids = {'player'}
        core.previous_state = [player, widget]
        core.update_state_display_evidence(frame, [player, widget], True)
        self.assertNotIn('widget', core.state_display_tracks)
        core.previous_state = [player, widget]
        core.update_state_display_evidence(frame, [player, widget], True)
        self.assertIn('widget', core.state_display_tracks)

    def test_failure_memory_does_not_penalize_identical_other_target(self):
        core = M['MyAgentCore']()
        objects = core.tracker.update([obj({(0, 0)}), obj({(0, 10)}, frame_id=1)])
        a, b = [o.track_id for o in objects]
        core.previous_state = objects
        core.goal_target_tracks = {'reach_object': [a], 'activate_object': [b]}
        core.current_goal = {'type': 'reach_object', 'target_tracks': [a]}
        core.record_goal_outcome('loop', True)
        self.assertIsNone(core.goal_memory_penalties()['reach_object'])
        self.assertNotIn('activate_object', core.goal_memory_penalties())
        after = core.tracker.update([obj({(0, 0)}, color=3), obj({(0, 10)}, frame_id=1)])
        core.previous_state = after
        self.assertIsNone(core.goal_memory_penalties()['reach_object'])

    def test_composite_goal_outcome_is_recorded_once(self):
        core = M['MyAgentCore']()
        objects = core.tracker.update([
            obj({(0, 0)}), obj({(0, 2)}, frame_id=1), obj({(0, 4)}, frame_id=2),
        ])
        tracks = [item.track_id for item in objects]
        core.previous_state = objects
        core.goal_target_tracks = {'reach_object': tracks}
        core.current_goal = {'type': 'reach_object', 'target_tracks': tracks}

        core.record_goal_outcome('target unresolved', False)

        self.assertEqual(len(core.goal_experiments), 1)
        record = next(iter(core.goal_experiments.values()))
        self.assertEqual(set(record['target_tracks']), set(tracks))
        self.assertEqual(record['inconclusive_count'], 1)
        self.assertEqual(core.goal_memory_penalties()['reach_object'], .05)

    def test_episode_namespaces_and_fresh_memory(self):
        a, b = M['MyAgentCore'](1), M['MyAgentCore'](2)
        x = a.tracker.update([obj({(0, 0)})])[0]
        y = b.tracker.update([obj({(0, 0)})])[0]
        self.assertNotEqual(x.track_id, y.track_id)
        self.assertFalse(b.goal_experiments)

    def test_discovery_uses_tracks_across_renumbered_frames(self):
        core = M['MyAgentCore']()
        frames = []
        for y, x in [(5, 5), (3, 5), (5, 5), (5, 3), (5, 5)]:
            frame = np.zeros((12, 12), dtype=int)
            frame[y:y+2, x:x+2] = 1
            frames.append(frame)
        core.observe(frames[0])
        for action, frame in zip(list(Action)[1:], frames[1:]):
            core.previous_action = action
            core.observe(frame)
        self.assertEqual(len(core.action_vectors), 4)
        self.assertEqual(len(core.controlled_track_ids), 1)
        self.assertEqual(len(core.controlled_component_ids), 1)

    def test_controlled_role_follows_split_and_merge_without_reusing_ids(self):
        core = M['MyAgentCore']()
        frame = np.zeros((10, 10), dtype=int)
        frame[3:5, 3:5] = 1
        objects = core.observe(frame)
        player = next(o for o in objects if o.color == 1)
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.mode = 'ACT'
        core.previous_action = Action.ACTION1
        core.action_vectors = {Action.ACTION1: (0, -2)}
        core.last_action_state = frozenset(player.pixels)
        frame[3, 3:5] = 2
        core.observe(frame)
        split_tracks = set(core.controlled_track_ids)
        self.assertEqual(len(split_tracks), 2)
        self.assertNotIn(player.track_id, split_tracks)
        self.assertEqual(len(M['get_controlled_pixels'](core.previous_state, core.controlled_component_ids)), 4)
        frame[3, 3:5] = 1
        core.observe(frame)
        self.assertEqual(len(core.controlled_track_ids), 1)
        self.assertFalse(core.controlled_track_ids & split_tracks)

    def test_ui_appearance_change_does_not_hide_blocked_move(self):
        core = M['MyAgentCore']()
        frame = np.zeros((12, 12), dtype=int)
        frame[3:5, 3:5] = 1
        frame[10, 10] = 2
        objects = core.observe(frame)
        player = next(o for o in objects if o.color == 1)
        ui = next(o for o in objects if o.color == 2)
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.scene_analysis = {'wall_candidates': [], 'ui_candidates': [ui.id]}
        core.capture_scene_role_tracks(objects)
        core.mode = 'ACT'
        core.previous_action = Action.ACTION1
        core.action_vectors = {Action.ACTION1: (0, -2)}
        core.last_action_state = frozenset(player.pixels)
        source = core.last_action_state
        frame[10, 10] = 3
        core.observe(frame)
        self.assertIn((source, Action.ACTION1), core.blocked_actions)

    def test_door_change_clears_stale_navigation_memory(self):
        core = M['MyAgentCore']()
        frame = np.zeros((12, 12), dtype=int)
        frame[3:5, 3:5] = 1
        frame[8, 8] = 2
        objects = core.observe(frame)
        player = next(o for o in objects if o.color == 1)
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.mode = 'ACT'
        core.previous_action = Action.ACTION1
        core.action_vectors = {Action.ACTION1: (0, -2)}
        core.last_action_state = frozenset(player.pixels)
        source = core.last_action_state
        core.failed_moves.add((source, Action.ACTION1))
        core.blocked_actions.add((source, Action.ACTION1))
        core.transition_graph[source][Action.ACTION1] = frozenset({(1, 1)})
        core.tested_actions[source].add(Action.ACTION1)
        core.state_visit_counts[source] = 2

        # A door opening changes a non-traversable gameplay component while
        # the player remains in place. The previous blocked result is stale.
        core.traversable_color_evidence[0] = 2
        frame[8, 8] = 0
        core.observe(frame)

        self.assertFalse(core.failed_moves)
        self.assertFalse(core.blocked_actions)
        self.assertFalse(core.transition_graph)
        self.assertFalse(core.tested_actions)
        # The reset occurs before the current position is recorded again.
        self.assertEqual(core.state_visit_counts, {source: 1})

    def test_floor_and_ui_changes_do_not_clear_navigation_memory(self):
        core = M['MyAgentCore']()
        frame = np.zeros((12, 12), dtype=int)
        frame[3:5, 3:5] = 1
        frame[8, 8] = 2
        objects = core.observe(frame)
        player = next(o for o in objects if o.color == 1)
        ui = next(o for o in objects if o.color == 2)
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.mode = 'ACT'
        core.previous_action = Action.ACTION1
        core.action_vectors = {Action.ACTION1: (0, -2)}
        core.last_action_state = frozenset(player.pixels)
        source = core.last_action_state
        core.failed_moves.add((source, Action.ACTION1))
        core.blocked_actions.add((source, Action.ACTION1))
        core.traversable_color_evidence[0] = 2
        core.scene_analysis = {'wall_candidates': [], 'ui_candidates': [ui.id]}
        core.capture_scene_role_tracks(objects)

        # A UI animation and the background reshaping around a stationary
        # player do not alter collision topology.
        frame[8, 8] = 3
        core.observe(frame)

        self.assertIn((source, Action.ACTION1), core.failed_moves)
        self.assertIn((source, Action.ACTION1), core.blocked_actions)

    def test_player_motion_does_not_treat_background_reshaping_as_a_door_change(self):
        core = M['MyAgentCore']()
        frame = np.zeros((12, 12), dtype=int)
        frame[3:5, 3:5] = 1
        objects = core.observe(frame)
        player = next(o for o in objects if o.color == 1)
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.mode = 'ACT'
        core.previous_action = Action.ACTION4
        core.action_vectors = {Action.ACTION1: (0, -2), Action.ACTION4: (2, 0)}
        core.last_action_state = frozenset(player.pixels)
        remembered = (core.last_action_state, Action.ACTION1)
        core.failed_moves.add(remembered)
        core.blocked_actions.add(remembered)

        frame[3:5, 3:5] = 0
        frame[3:5, 5:7] = 1
        core.observe(frame)

        self.assertIn(remembered, core.failed_moves)
        self.assertIn(remembered, core.blocked_actions)

    def test_target_contact_records_ui_effect_and_requests_reanalysis(self):
        core = M['MyAgentCore']()
        player = obj({(5, 1), (5, 2)}, color=8, frame_id=1)
        target = obj({(4, 1)}, color=0, frame_id=2)
        ui = obj({(0, 6)}, color=11, frame_id=3)
        before = core.tracker.update([player, target, ui])
        player, target, ui = before
        core.previous_state = before
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(before)
        core.current_goal = {
            'type': 'reach_object', 'target_tracks': [target.track_id],
        }
        core.scene_analysis = {'ui_candidates': [ui.id]}
        core.pending_interaction = core.begin_interaction_observation([target.id])
        core.previous_action = Action.ACTION1

        after = core.tracker.update(
            [obj({(4, 1), (4, 2)}, color=8, frame_id=7),
             obj({(0, 6)}, color=12, frame_id=8)],
            core.controlled_track_ids,
            (0, -1),
        )
        core.resolve_controlled_ids(after)
        observation = core.observe_pending_interaction(after)

        self.assertTrue(observation['contacted'])
        self.assertEqual(observation['target_not_visible'], [target.track_id])
        self.assertEqual(observation['ui_changes'][0]['track_id'], ui.track_id)
        self.assertEqual(observation['association_confidence'], 'strong_single_observation')
        self.assertTrue(core.needs_reanalysis)
        self.assertIn('target contact produced', core.reanalysis_reason)
        self.assertFalse(core.goal_experiments)
        self.assertIsNone(core.pending_interaction)

    def test_contact_provides_only_changed_or_new_gameplay_candidates(self):
        core = M['MyAgentCore']()
        before_frame = np.full((20, 20), 3, dtype=int)
        player = obj({(7, 1), (7, 2)}, color=1, frame_id=1)
        target = obj({(6, 1)}, color=0, frame_id=2)
        changed = obj({(5, 5)}, color=2, frame_id=3)
        stable = obj({(8, 2)}, color=2, frame_id=4)
        ui = obj({(0, 8)}, color=11, frame_id=5)
        unlocked = obj({(6, 8)}, color=7, frame_id=6)
        barrier = obj({(row, 4) for row in range(20)}, color=5, frame_id=7)
        for item in (player, target, changed, stable, ui, unlocked, barrier):
            for y, x in item.pixels:
                before_frame[y, x] = item.color
        before = core.tracker.update([player, target, changed, stable, ui, unlocked, barrier])
        player, target, changed, stable, ui, unlocked, barrier = before
        core.previous_state = before
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(before)
        core.current_goal = {'type': 'reach_object', 'target_tracks': [target.track_id]}
        core.scene_analysis = {'wall_candidates': [], 'ui_candidates': [ui.id]}
        core.capture_scene_role_tracks(before)
        core.traversable_color_evidence[3] = 2
        core.action_vectors = {
            Action.ACTION1: (0, -1), Action.ACTION2: (0, 1),
            Action.ACTION3: (-1, 0), Action.ACTION4: (1, 0),
        }
        core.previous_action = Action.ACTION1
        core.pending_interaction = core.begin_interaction_observation(
            [target.id], Action.ACTION1, frame=before_frame,
        )

        after_frame = np.full((20, 20), 3, dtype=int)
        after_objects = [
            obj({(6, 1), (6, 2)}, color=1, frame_id=10),
            obj({(5, 5)}, color=4, frame_id=11),
            obj({(8, 2)}, color=2, frame_id=12),
            obj({(0, 8)}, color=12, frame_id=13),
            obj({(10, 12)}, color=6, frame_id=14),
            obj({(6, 8)}, color=7, frame_id=15),
        ]
        for item in after_objects:
            for y, x in item.pixels:
                after_frame[y, x] = item.color
        after = core.tracker.update(
            after_objects, core.controlled_track_ids, (0, -1),
        )
        core.resolve_controlled_ids(after)
        observation = core.observe_pending_interaction(after, frame=after_frame)

        by_track = {item['track_id']: item for item in observation['next_goal_candidates']}
        changed_after = next(o for o in after if o.track_id == changed.track_id)
        new_after = next(o for o in after if o.color == 6)
        unlocked_after = next(o for o in after if o.track_id == unlocked.track_id)
        self.assertIn(changed_after.track_id, by_track)
        self.assertIn('changed', by_track[changed_after.track_id]['reasons'])
        self.assertIn(new_after.track_id, by_track)
        self.assertIn('appeared', by_track[new_after.track_id]['reasons'])
        self.assertIn(unlocked_after.track_id, by_track)
        self.assertIn('became_reachable', by_track[unlocked_after.track_id]['reasons'])
        self.assertNotIn(target.track_id, by_track)
        self.assertNotIn(stable.track_id, by_track)
        self.assertNotIn(ui.track_id, by_track)
        self.assertEqual(core.post_interaction_candidates, observation['next_goal_candidates'])
        self.assertEqual(observation['interaction_classification'], 'route_opened')
        self.assertIsNone(core.repeat_activation_plan)

    def test_repeatable_interaction_records_rotation_and_stops_on_state_cycle(self):
        core = M['MyAgentCore']()
        original = obj({(0, 0), (1, 0), (1, 1)}, color=6, frame_id=1)
        rotated = obj({(0, 0), (0, 1), (1, 0)}, color=6, frame_id=1)
        original.track_id = rotated.track_id = 'pattern'
        initial_state, _ = core.interaction_state_signature([original], {'switch'})
        pending = {
            'goal_type': 'activate_object', 'target_tracks': ('switch',),
            'target_region_pixels': frozenset({(5, 5)}),
            'state_signature': initial_state,
            'snapshots': {'pattern': core.snapshot_track(original)},
        }
        changed = [{'track_id': 'pattern', 'kind': 'changed', 'area': 'ui',
                    'fields': ['shape_hash']}]
        core.post_interaction_candidates = []
        first = core.record_interaction_transition(
            [rotated], pending, changed, {'pattern': rotated},
        )
        self.assertEqual(first['classification'], 'repeatable_pending')
        self.assertEqual(
            first['last_transition']['transforms'][0]['transform'], 'rotation_cw_90',
        )
        core.repeat_activation_plan['phase'] = 'reenter'
        core.confirm_repeat_activation_state([rotated])
        self.assertTrue(core.repeat_activation_plan['persistence_confirmed'])
        self.assertEqual(first['classification'], 'repeatable_awaiting_relation')
        # A pattern display alone is not sufficient to revisit its actuator.
        # A later reanalysis must link it to a real gameplay target.
        core.repeat_activation_plan['relation_confirmed'] = True
        core.confirm_repeat_activation_state([rotated])
        self.assertEqual(first['classification'], 'repeatable')

        rotated_state, _ = core.interaction_state_signature([rotated], {'switch'})
        return_pending = dict(
            pending,
            state_signature=rotated_state,
            snapshots={'pattern': core.snapshot_track(rotated)},
        )
        second = core.record_interaction_transition(
            [original], return_pending, changed, {'pattern': original},
        )
        self.assertEqual(second['classification'], 'state_cycle')
        self.assertIsNone(core.repeat_activation_plan)

    def test_repeatable_activation_exits_then_reenters_target(self):
        core = M['MyAgentCore']()
        frame = np.full((12, 12), 3, dtype=int)
        player = obj({(5, 5)}, color=1, frame_id=1)
        target = obj({(5, 5)}, color=0, frame_id=2)
        player.track_id, target.track_id = 'player', 'switch'
        core.previous_state = [player, target]
        core.controlled_track_ids = {'player'}
        core.controlled_component_ids = {player.id}
        core.traversable_color_evidence[3] = 2
        actions = {Action.ACTION1: (0, -1), Action.ACTION2: (0, 1)}
        core.action_vectors = actions
        core.repeat_activation_plan = {
            'target_tracks': ('switch',), 'target_region_pixels': frozenset(target.pixels),
            'phase': 'exit', 'reentry_failures': 0,
            'expected_state_signature': 'state', 'persistence_confirmed': True,
            'relation_confirmed': True,
        }
        self.assertEqual(core.choose_repeat_activation_action(frame, actions), Action.ACTION1)
        self.assertEqual(core.repeat_activation_plan['phase'], 'reenter')

        player.pixels = {(4, 5)}
        player.bbox = (4, 5, 4, 5)
        player.center = (4.0, 5.0)
        core.previous_state = [player, target]
        self.assertEqual(core.choose_repeat_activation_action(frame, actions), Action.ACTION2)
        self.assertEqual(core.current_goal['target_tracks'], ['switch'])
        self.assertIsNotNone(core.pending_interaction)

    def test_display_change_does_not_reenter_without_goal_relation(self):
        core = M['MyAgentCore']()
        frame = np.full((12, 12), 3, dtype=int)
        player = obj({(4, 5)}, color=1, frame_id=1)
        target = obj({(5, 5)}, color=0, frame_id=2)
        player.track_id, target.track_id = 'player', 'cross'
        core.previous_state = [player, target]
        core.controlled_track_ids = {'player'}
        core.controlled_component_ids = {player.id}
        core.traversable_color_evidence[3] = 2
        core.repeat_activation_plan = {
            'target_tracks': ('cross',), 'target_region_pixels': frozenset(target.pixels),
            'phase': 'reenter', 'reentry_failures': 0,
            'expected_state_signature': 'state', 'persistence_confirmed': True,
            'requires_goal_relation': True, 'relation_confirmed': False,
        }
        self.assertIsNone(core.choose_repeat_activation_action(
            frame, {Action.ACTION2: (0, 1)},
        ))
        core.goal_target_tracks = {'activate_object': ['cross'], 'reach_object': ['goal']}
        core.current_goal = {'type': 'activate_object', 'target_tracks': ['cross']}
        self.assertIsNone(core.goal_memory_penalties()['activate_object'])

    def test_match_pattern_analysis_approves_display_repeat_for_world_target(self):
        core = M['MyAgentCore']()
        core.repeat_activation_plan = {
            'target_tracks': ('cross',), 'requires_goal_relation': True,
            'relation_confirmed': False,
        }
        core.scene_analysis = {
            'goal_scores': {'match_pattern': {'confidence': .8, 'target_ids': [7]}},
        }
        core.goal_target_tracks = {'match_pattern': ['world-goal']}
        core.interaction_transitions = {
            ('move_to_target', ('cross',)): {'classification': 'repeatable_pending'},
        }
        core.approve_repeat_activation_after_analysis()
        self.assertTrue(core.repeat_activation_plan['relation_confirmed'])
        self.assertEqual(
            core.interaction_transitions[('move_to_target', ('cross',))]['relation_confirmed'],
            True,
        )

    def test_gameplay_switch_transition_can_repeat_without_pattern_relation(self):
        core = M['MyAgentCore']()
        before = obj({(1, 1)}, color=6, frame_id=1)
        after = obj({(1, 2)}, color=6, frame_id=1)
        before.track_id = after.track_id = 'door-state'
        state, _ = core.interaction_state_signature([before], {'switch'})
        record = core.record_interaction_transition(
            [after],
            {
                'goal_type': 'activate_object', 'target_tracks': ('switch',),
                'target_region_pixels': frozenset({(5, 5)}),
                'state_signature': state,
                'snapshots': {'door-state': core.snapshot_track(before)},
            },
            [{'track_id': 'door-state', 'kind': 'changed', 'area': 'gameplay',
              'fields': ['bbox']}],
            {'door-state': after},
        )
        self.assertEqual(record['classification'], 'repeatable_pending')
        self.assertFalse(core.repeat_activation_plan['requires_goal_relation'])
        self.assertTrue(core.repeat_activation_plan['relation_confirmed'])

    def test_disappearing_observed_target_is_one_shot_not_repeatable(self):
        core = M['MyAgentCore']()
        target = obj({(5, 5)}, color=6, frame_id=1)
        display = obj({(0, 1), (1, 1)}, color=9, frame_id=2)
        target.track_id, display.track_id = 'cross', 'display'
        state, _ = core.interaction_state_signature([target, display], {'cross'})
        record = core.record_interaction_transition(
            [display],
            {
                'goal_type': 'activate_object', 'target_tracks': ('cross',),
                'target_region_pixels': frozenset(target.pixels),
                'state_signature': state,
                'snapshots': {
                    'cross': core.snapshot_track(target),
                    'display': core.snapshot_track(display),
                },
            },
            [{'track_id': 'display', 'kind': 'changed', 'area': 'ui',
              'fields': ['shape_hash']}],
            {'display': display},
        )
        self.assertEqual(record['classification'], 'one_shot_transition')
        self.assertIsNone(core.repeat_activation_plan)

    def test_untrusted_target_is_planned_to_boundary_not_entered(self):
        frame = np.zeros((10, 10), dtype=int)
        player = obj({(5, 3)}, color=1, frame_id=1)
        target = obj({(5, 4)}, color=5, frame_id=2)
        action = M['choose_goal_action_bfs'](
            frame, [player, target], {player.id}, [target.id],
            {Action.ACTION4: (1, 0)}, {0: 2}, set(),
            allow_target_overlap=False,
        )
        self.assertIsNone(action)  # Already at a legal contact boundary.
        overlap_action = M['choose_goal_action_bfs'](
            frame, [player, target], {player.id}, [target.id],
            {Action.ACTION4: (1, 0)}, {0: 2}, set(),
            allow_target_overlap=True,
        )
        self.assertEqual(overlap_action, Action.ACTION4)

    def test_ambiguous_pattern_hypothesis_does_not_materialize_navigation_goal(self):
        core = M['MyAgentCore']()
        left = obj({(2, 2)}, color=9, frame_id=3)
        right = obj({(2, 8)}, color=9, frame_id=4)
        left.track_id, right.track_id = 'left', 'right'
        core.scene_analysis = {
            'goal_scores': {
                'reach_object': {'confidence': 0, 'target_ids': [], 'evidence': ''},
                'match_pattern': {'confidence': .9, 'target_ids': [3, 4], 'evidence': ''},
            },
            'pattern_hypothesis': {
                'condition_status': 'matched',
                'required_next_step': 'navigate_world_target',
                'world_pattern_ids': [3, 4], 'navigation_target_ids': [3, 4],
            },
        }
        core.pattern_relation_context = {
            'gameplay_pattern_clusters': [
                {'track_ids': ['left']}, {'track_ids': ['right']},
            ],
        }
        core.materialize_pattern_navigation_goal([left, right])
        self.assertEqual(core.scene_analysis['goal_scores']['reach_object']['target_ids'], [])

    def test_match_pattern_score_is_not_an_executable_goal_without_materialization(self):
        scene = {
            'goal_scores': {
                'match_pattern': {'confidence': .9, 'target_ids': [7]},
                'reach_object': {'confidence': .2, 'target_ids': [8]},
                'unknown': {'confidence': .1, 'target_ids': []},
            },
        }
        selected = M['select_primary_goal'](
            scene, excluded_goal_types={'match_pattern'},
        )
        self.assertIsNone(selected)

    def test_pattern_context_uses_changed_compact_shapes_without_raw_pixel_payloads(self):
        core = M['MyAgentCore']()
        frame = np.zeros((20, 20), dtype=int)
        panel = obj({(y, x) for y in range(10) for x in range(10)}, color=5, frame_id=1)
        display = obj({(2, 2), (2, 3), (3, 2)}, color=9, frame_id=2)
        unchanged = obj({(4, 2)}, color=9, frame_id=3)
        goal = obj({(12, 12), (12, 13), (13, 12)}, color=9, frame_id=4)
        panel.track_id, display.track_id = 'panel', 'display'
        unchanged.track_id, goal.track_id = 'unchanged', 'goal'
        core.state_display_tracks = {'panel', 'display', 'unchanged'}
        core.pattern_display_tracks = {'display'}
        core.pattern_gameplay_tracks = {'goal'}
        context = core.build_pattern_relation_context(frame, [panel, display, unchanged, goal])
        self.assertEqual([item['track_id'] for item in context['state_displays']], ['display'])
        self.assertEqual(context['gameplay_shape_candidates'][0]['track_id'], 'goal')
        cluster = context['state_display_clusters'][0]
        self.assertNotIn('pixels_by_color', cluster)
        self.assertIn('shape_rle', cluster)
        self.assertIn('color_counts', cluster)
        self.assertNotIn('panel', cluster['track_ids'])

    def test_reset_memory_holds_visual_equivalent_goal_after_rebuild(self):
        memory = {'unsafe_goal_signatures': set(), 'reset_events': []}
        first = M['MyAgentCore'](0, level_memory=memory)
        target = obj({(4, 4)}, color=8, frame_id=1)
        target.track_id = 'danger'
        first.previous_state = [target]
        first.current_goal = {'type': 'reach_object', 'target_tracks': ['danger']}
        first.previous_action = Action.ACTION1
        first.record_reset_risk()
        second = M['MyAgentCore'](1, level_memory=memory)
        replacement = obj({(4, 4)}, color=8, frame_id=9)
        replacement.track_id = 'replacement'
        second.previous_state = [replacement]
        second.goal_target_tracks = {'reach_object': ['replacement']}
        self.assertIsNone(second.goal_memory_penalties()['reach_object'])

    def test_two_no_effect_contacts_stop_repeatable_hypothesis(self):
        core = M['MyAgentCore']()
        widget = obj({(1, 1)}, color=6, frame_id=1)
        widget.track_id = 'widget'
        state, _ = core.interaction_state_signature([widget], {'switch'})
        pending = {
            'goal_type': 'activate_object', 'target_tracks': ('switch',),
            'target_region_pixels': frozenset({(5, 5)}),
            'state_signature': state,
            'snapshots': {'widget': core.snapshot_track(widget)},
        }
        core.post_interaction_candidates = []
        first = core.record_interaction_transition(
            [widget], pending, [], {'widget': widget},
        )
        second = core.record_interaction_transition(
            [widget], pending, [], {'widget': widget},
        )
        self.assertEqual(first['no_effect_count'], 2)
        self.assertEqual(second['classification'], 'stalled')
        self.assertIsNone(core.repeat_activation_plan)

    def test_predicted_contact_survives_target_and_player_resegmentation(self):
        core = M['MyAgentCore']()
        player = obj({(5, 1), (5, 2)}, color=8, frame_id=1)
        target = obj({(4, 1)}, color=0, frame_id=2)
        before = core.tracker.update([player, target])
        player, target = before
        core.previous_state = before
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(before)
        core.current_goal = {
            'type': 'reach_object', 'target_tracks': [target.track_id],
        }
        core.scene_analysis = {'ui_candidates': []}
        core.action_vectors = {Action.ACTION1: (0, -1)}
        core.previous_action = Action.ACTION1
        core.pending_interaction = core.begin_interaction_observation(
            [target.id], Action.ACTION1,
        )

        # A consumed marker can cause the tracker to lose both the target and
        # the old controlled component in the contact frame.
        after = core.tracker.update([], core.controlled_track_ids, (0, -1))
        core.controlled_component_ids = set()
        observation = core.observe_pending_interaction(
            after, movement_succeeded=True,
        )

        self.assertTrue(observation['contacted'])
        self.assertEqual(observation['contact_method'], 'predicted')
        self.assertEqual(observation['target_not_visible'], [target.track_id])
        self.assertTrue(core.needs_reanalysis)

    def test_blocked_planned_contact_does_not_create_causal_evidence(self):
        core = M['MyAgentCore']()
        player = obj({(5, 1), (5, 2)}, color=8, frame_id=1)
        target = obj({(4, 1)}, color=0, frame_id=2)
        before = core.tracker.update([player, target])
        player, target = before
        core.previous_state = before
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(before)
        core.current_goal = {
            'type': 'reach_object', 'target_tracks': [target.track_id],
        }
        core.scene_analysis = {'ui_candidates': []}
        core.action_vectors = {Action.ACTION1: (0, -1)}
        core.previous_action = Action.ACTION1
        core.pending_interaction = core.begin_interaction_observation(
            [target.id], Action.ACTION1,
        )

        self.assertIsNone(core.observe_pending_interaction(
            before, movement_succeeded=False,
        ))
        self.assertFalse(core.mechanics_evidence)
        self.assertFalse(core.needs_reanalysis)

    def test_ui_change_without_target_contact_is_not_associated(self):
        core = M['MyAgentCore']()
        player = obj({(5, 1), (5, 2)}, color=8, frame_id=1)
        target = obj({(4, 1)}, color=0, frame_id=2)
        ui = obj({(0, 6)}, color=11, frame_id=3)
        before = core.tracker.update([player, target, ui])
        player, target, ui = before
        core.previous_state = before
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(before)
        core.current_goal = {
            'type': 'reach_object', 'target_tracks': [target.track_id],
        }
        core.scene_analysis = {'ui_candidates': [ui.id]}
        core.pending_interaction = core.begin_interaction_observation([target.id])
        core.previous_action = Action.ACTION3

        after = core.tracker.update(
            [obj({(5, 0), (5, 1)}, color=8, frame_id=7),
             obj({(4, 1)}, color=0, frame_id=2),
             obj({(0, 6)}, color=12, frame_id=8)],
            core.controlled_track_ids,
            (-1, 0),
        )
        core.resolve_controlled_ids(after)
        self.assertIsNone(core.observe_pending_interaction(after))
        self.assertFalse(core.mechanics_evidence)
        self.assertFalse(core.needs_reanalysis)

    def test_repeated_effect_strengthens_mechanics_evidence(self):
        core = M['MyAgentCore']()
        observation = {
            'goal_type': 'activate_object', 'target_tracks': ['switch'],
            'target_not_visible': [],
            'ui_changes': [{'track_id': 'indicator', 'kind': 'changed', 'area': 'ui'}],
            'gameplay_changes': [],
        }
        self.assertEqual(
            core.record_mechanics_observation(dict(observation))['association_confidence'],
            'observed_once',
        )
        self.assertEqual(
            core.record_mechanics_observation(dict(observation))['association_confidence'],
            'repeated',
        )
        feedback = core.cumulative_mechanics_feedback()
        self.assertEqual(feedback[0]['contact_count'], 2)
        self.assertEqual(feedback[0]['association_confidence'], 'repeated')

    def test_runner_reinitializes_on_reset_and_level_change(self):
        states = [
            SimpleNamespace(raw=SimpleNamespace(state=state, levels_completed=level),
                            frame=SimpleNamespace(data=np.zeros((2, 2))), available_actions=[1])
            for state, level in [('playing', 0), ('playing', 1), ('over', 1), ('playing', 0)]
        ]
        constructed, used, actions = [], [], []

        def core_factory(episode_id=0):
            constructed.append(episode_id)
            def choose_action(**kwargs):
                used.append(episode_id)
                return Action.ACTION1
            return SimpleNamespace(choose_action=choose_action)

        class Game:
            game_run = SimpleNamespace(state='playing', final_score=None)
            current_state = states[0]

            def execute_action(self, action):
                actions.append(action.id)
                if len(actions) == len(states):
                    self.game_run.state = 'finished'
                else:
                    self.current_state = states[len(actions)]

            def finish_game(self):
                self.game_run.final_score = 0

        with patch.dict(M, MyAgentCore=core_factory):
            asyncio.run(M['MyAgentSolver']()._play_one(Game()))
        self.assertEqual(constructed, [0, 1, 2])
        self.assertEqual(used, [0, 1, 2])
        self.assertEqual(actions, [Action.ACTION1, Action.ACTION1, Action.RESET, Action.ACTION1])

    def test_runner_finishes_on_engine_win_without_another_action(self):
        states = [
            SimpleNamespace(raw=SimpleNamespace(state=state, levels_completed=0),
                            frame=SimpleNamespace(data=np.zeros((2, 2))), available_actions=[1])
            for state in ('playing', 'win')
        ]
        actions, contexts, finish_calls = [], [], []

        def core_factory(episode_id=0):
            def choose_action(**kwargs):
                return Action.ACTION1
            def set_runtime_context(raw_state, game_run):
                contexts.append((raw_state.state, raw_state.levels_completed, game_run.state))
            return SimpleNamespace(
                choose_action=choose_action,
                set_runtime_context=set_runtime_context,
            )

        class Game:
            game_run = SimpleNamespace(state='playing', final_score=None)
            current_state = states[0]

            def execute_action(self, action):
                actions.append(action.id)
                self.current_state = states[1]

            def finish_game(self):
                finish_calls.append(True)
                self.game_run.final_score = 1

        with patch.dict(M, MyAgentCore=core_factory):
            asyncio.run(M['MyAgentSolver']()._play_one(Game()))
        self.assertEqual(actions, [Action.ACTION1])
        self.assertEqual(contexts, [('playing', 0, 'playing')])
        self.assertEqual(finish_calls, [True])

    def test_runtime_context_is_passed_to_scene_analysis(self):
        core = M['MyAgentCore']()
        frame = np.zeros((8, 8), dtype=int)
        frame[3:5, 3:5] = 1
        objects = core.observe(frame)
        player = next(o for o in objects if o.color == 1)
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.mode = 'ANALYZE'
        core.set_runtime_context(
            SimpleNamespace(state='playing', levels_completed=2),
            SimpleNamespace(state='playing'),
        )
        core.post_interaction_candidates = [
            {'frame_id': 4, 'track_id': 'new-door', 'reasons': ['appeared']},
        ]
        core.interaction_transitions = {
            ('move_to_target', ('switch',)): {
                'target_tracks': ('switch',), 'classification': 'repeatable',
                'state_signatures': {'state-a', 'state-b'}, 'no_effect_count': 0,
                'cycle_detected': False,
                'last_transition': {'transforms': [{'transform': 'rotation_cw_90'}]},
            },
        }
        analysis = {
            'wall_candidates': [], 'ui_candidates': [], 'important_objects': [],
            'goal_scores': {'unknown': {'confidence': 1, 'target_ids': []}},
        }
        captured = {}
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(
            M, analyze_scene_vlm=lambda **kwargs: captured.update(kwargs) or analysis,
        ):
            core.choose_action(frame, [1])
        self.assertEqual(captured['runtime_context'], {
            'engine_state': 'playing', 'levels_completed': 2, 'run_state': 'playing',
        })
        self.assertEqual(captured['post_interaction_candidates'], core.post_interaction_candidates)
        self.assertEqual(captured['interaction_transitions'][0]['classification'], 'repeatable')
        self.assertEqual(captured['interaction_transitions'][0]['states_seen'], 2)
        self.assertIn('state_displays', captured['pattern_relation_context'])

    def test_analysis_to_action_uses_track_targets(self):
        core = M['MyAgentCore']()
        frame = np.zeros((12, 12), dtype=int)
        frame[5:7, 3:5] = 1
        frame[5:7, 7:9] = 2
        objects = core.observe(frame)
        player = next(o for o in objects if o.color == 1)
        target = next(o for o in objects if o.color == 2)
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.action_vectors = {Action.ACTION1: (0, -2), Action.ACTION2: (0, 2),
                               Action.ACTION3: (-2, 0), Action.ACTION4: (2, 0)}
        core.traversable_color_evidence[0] = 2
        core.mode = 'ANALYZE'
        analysis = {'wall_candidates': [], 'ui_candidates': [], 'important_objects': [target.id],
                    'goal_scores': {'unknown': {'confidence': 0, 'target_ids': []},
                                    'reach_object': {'confidence': .9, 'target_ids': [target.id]}}}
        captured = {}
        def analyze_scene_vlm(**kwargs):
            captured.update(kwargs)
            return analysis
        core.record_mechanics_observation({
            'goal_type': 'activate_object', 'target_tracks': ['prior_switch'],
            'target_not_visible': [],
            'ui_changes': [{'track_id': 'indicator', 'kind': 'changed', 'area': 'ui'}],
            'gameplay_changes': [],
        })
        with patch.dict(M, analyze_scene_vlm=analyze_scene_vlm):
            action = core.choose_action(frame, [1, 2, 3, 4])
        self.assertEqual(action, Action.ACTION4)
        self.assertEqual(core.current_goal['target_tracks'], [target.track_id])
        self.assertEqual(captured['causal_observations'][0]['target_tracks'], ['prior_switch'])
        frame[5:7, 3:5] = 0
        frame[5:7, 5:7] = 1
        self.assertEqual(core.choose_action(frame, [4]), Action.ACTION4)
        self.assertEqual(core.current_goal['target_tracks'], [target.track_id])


if __name__ == '__main__':
    unittest.main()
