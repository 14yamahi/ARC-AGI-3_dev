"""Pure tracking regressions, runnable without Kaggle or a model endpoint.

Load the solver's definitions without its import-time HTTP client and optional
Kaggle imports. The production implementations themselves are exercised.
"""
import ast
import asyncio
import contextlib
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
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
        'select_primary_goal', 'MyAgentSolver',
        'validate_scene_analysis',
        'choose_goal_action_bfs', 'choose_frontier_action', 'choose_exploration_action',
        'plan_to_nearest_frontier', 'get_frontier_actions', 'inverse_action',
    }
    namespace = dict(dataclass=dataclass, np=np, hashlib=hashlib,
                     deque=deque, defaultdict=defaultdict,
                     arcengine=SimpleNamespace(GameAction=Action, GameState=SimpleNamespace(GAME_OVER='over'),
                                               ActionInput=lambda **kw: SimpleNamespace(**kw)),
                     taaf=SimpleNamespace(game=SimpleNamespace(Game=object)), Solver=object,
                     asyncio=asyncio, json=json, contextlib=contextlib, traceback=traceback)
    namespace['normalize_scene_analysis'] = lambda analysis: analysis
    exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, 'name', None) in names],
                            type_ignores=[]), str(path), 'exec'), namespace)
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

    def test_validation_rejects_ambiguous_target(self):
        objects = [obj({(0, 0)}, frame_id=5)]
        analysis = {'wall_candidates': [], 'ui_candidates': [], 'important_objects': [5],
                    'goal_scores': {'reach_object': {'target_ids': [5], 'confidence': .9, 'evidence': ''}}}
        result = M['validate_scene_analysis'](analysis, set(), objects, {})
        self.assertEqual(result['goal_scores']['reach_object']['target_ids'], [])
        self.assertEqual(result['goal_scores']['reach_object']['confidence'], 0)

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
        core.controlled_track_ids = {player.track_id}
        core.resolve_controlled_ids(objects)
        core.mode = 'ACT'
        core.previous_action = Action.ACTION1
        core.action_vectors = {Action.ACTION1: (0, -2)}
        core.last_action_state = frozenset(player.pixels)
        source = core.last_action_state
        frame[10, 10] = 3
        core.observe(frame)
        self.assertIn((source, Action.ACTION1), core.blocked_actions)

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
        with patch.dict(M, analyze_scene_vlm=lambda **kw: analysis):
            action = core.choose_action(frame, [1, 2, 3, 4])
        self.assertEqual(action, Action.ACTION4)
        self.assertEqual(core.current_goal['target_tracks'], [target.track_id])
        frame[5:7, 3:5] = 0
        frame[5:7, 5:7] = 1
        self.assertEqual(core.choose_action(frame, [4]), Action.ACTION4)
        self.assertEqual(core.current_goal['target_tracks'], [target.track_id])


if __name__ == '__main__':
    unittest.main()
