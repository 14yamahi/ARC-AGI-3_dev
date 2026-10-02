from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.my_agent3_test_support import MemoryLogger, load_solver

M = load_solver()


@pytest.fixture
def runtime():
    board = np.zeros((64, 64), dtype=np.int16)
    game = SimpleNamespace(
        current_state=SimpleNamespace(frame=SimpleNamespace(data=board),
            raw=SimpleNamespace(state=M.arcengine.GameState.NOT_FINISHED, levels_completed=0),
            available_actions=[M.arcengine.GameAction.ACTION1, M.arcengine.GameAction.ACTION6]),
        game_run=SimpleNamespace(state="playing", final_score=None),
    )
    game.calls = []
    def execute(action, **usage):
        game.calls.append(action)
        board[action.data.get("y", 0), action.data.get("x", 0)] = 1
    game.execute_action = execute
    return M.PythonToolRuntime(game, 22, MemoryLogger())


def test_coordinate_click_reaches_engine_and_mouse_alias_matches(runtime):
    runtime.execute("action([{'action': 'ACTION6', 'row': 4, 'col': 7}])")
    assert len(runtime.game.calls) == 1
    assert runtime.game.calls[0].data == {"x": 7, "y": 4}
    assert runtime._parse_action({"action": "MOUSE", "data": {"x": 8, "y": 9}})[0].name == "ACTION6"


@pytest.mark.parametrize("click_request", ["ACTION6", {"action": "ACTION6", "x": 1},
    {"action": "ACTION6", "x": -1, "y": 1}, {"action": "ACTION6", "x": 64, "y": 1},
    {"action": "ACTION6", "x": True, "y": 1}, {"action": "ACTION6", "row": 1.2, "col": 1}])
def test_bad_click_returns_feedback_without_touching_engine(runtime, click_request):
    result = runtime.execute(f"action([{click_request!r}])")
    assert "ERROR" in result
    assert runtime.game.calls == []
    assert runtime.actions_taken == 0


def test_observed_python_builtins_and_imports_are_available(runtime):
    result = runtime.execute("import numpy as np\nimport hashlib\ntry:\n action(['ACTION6'])\nexcept Exception as e:\n print(type(e), isinstance(np.zeros(2), np.ndarray), hasattr(current_frame, 'shape'), last_error)\n")
    assert "ValueError" in result
    assert "True True None" in result
    assert not runtime.game.calls


def test_prompt_describes_actual_action_result_and_crop_schema():
    assert "r[0]['change']" in M.SYSTEM_PROMPT
    assert "splitlines()" in M.SYSTEM_PROMPT
    assert 'action([{"action": "ACTION6", "row": 4, "col": 7}])' in M.SYSTEM_PROMPT
