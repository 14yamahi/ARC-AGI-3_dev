import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.my_agent3_test_support import load_solver


def validate():
    path = Path(__file__).resolve().parents[2] / 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb'
    notebook = json.loads(path.read_text())
    tree = ast.parse(''.join(notebook['cells'][16]['source']))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_validate_offline_runs')
    ns = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), ns)
    return ns['_validate_offline_runs']


def run(game_id='ls20', state='gave_up', final_score=0, actions=3, note=None):
    return SimpleNamespace(game_id=game_id, state=state, final_score=final_score, history=[None] * actions, solver_note=note)


def test_completed_zero_score_is_a_valid_execution():
    assert validate()([run()], ['ls20'], []) == 3


@pytest.mark.parametrize('bad_run', [run(state='playing'), run(state='cancelled'), run(state='crashed'),
                                   run(final_score=None), run(actions=0), run(note='RuntimeError: engine failed')])
def test_incomplete_or_failed_run_cannot_validate(bad_run):
    with pytest.raises(RuntimeError):
        validate()([bad_run], ['ls20'], [])


def test_wrong_coverage_and_duplicates_cannot_validate():
    for runs in ([], [run('cn04')], [run(), run()]):
        with pytest.raises(RuntimeError, match='coverage'):
            validate()(runs, ['ls20'], [])


def test_http_errors_cannot_be_hidden_by_gave_up_state():
    with pytest.raises(RuntimeError, match='failures'):
        validate()([run()], ['ls20'], [{'game_id': 'ls20', 'error': 'HTTP 400: context overflow'}])


def test_solver_records_model_failures_separately_from_game_score():
    solver = load_solver().MyAgent3Solver()
    solver._record_model_failure(SimpleNamespace(game_id='ls20'), 'preflight failed')
    assert solver._model_failures == [{'game_id': 'ls20', 'error': 'preflight failed'}]


def test_startup_removes_stale_offline_artifacts(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb'
    notebook = json.loads(path.read_text())
    setup = ast.parse(''.join(notebook['cells'][3]['source']))
    cleanup = next(node for node in setup.body if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                   and isinstance(node.test.operand, ast.Name) and node.test.operand.id == 'TRUE_SUBMISSION')
    for name in ('submission.parquet', 'offline_validation.json'):
        (tmp_path / name).write_text('old')
    exec(compile(ast.Module(body=[cleanup], type_ignores=[]), '<startup>', 'exec'),
         {'TRUE_SUBMISSION': False, 'WORKING_DIR': tmp_path})
    assert not (tmp_path / 'submission.parquet').exists()
    assert not (tmp_path / 'offline_validation.json').exists()


@pytest.mark.asyncio
async def test_failed_benchmark_does_not_create_placeholder_and_still_stops_watchdog(tmp_path, monkeypatch):
    from unittest.mock import Mock

    path = Path(__file__).resolve().parents[2] / 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb'
    notebook = json.loads(path.read_text())
    tree = ast.parse(''.join(notebook['cells'][16]['source']))
    execution = next(node for node in tree.body if isinstance(node, ast.Try))
    (tmp_path / 'teardown_commands.json').write_text('[]')
    async def play(**kwargs):
        pass
    watchdog = SimpleNamespace(stop_background=Mock())
    ns = {'bm': SimpleNamespace(run=play, _save_json=Mock(), solver=SimpleNamespace(_model_failures=[{'error': 'HTTP 400'}])),
          'soft_end': None, 'target': None, 'TRUE_SUBMISSION': False, 'WORKING_DIR': tmp_path,
          'BUNDLE_DIR': tmp_path, 'json': json, 'vllm_watchdog': watchdog, 'sys': sys}
    code = compile(ast.Module(body=[execution], type_ignores=[]), '<benchmark>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with pytest.raises(RuntimeError, match='Model failures'):
        await eval(code, ns)
    watchdog.stop_background.assert_called_once()
    assert not (tmp_path / 'submission.parquet').exists()
