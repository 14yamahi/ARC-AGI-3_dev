import ast
import asyncio
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from openai.types.chat import ChatCompletion

from scripts import duck_serving_teardown as cleanup
from tests.unit.my_agent3_test_support import MemoryLogger, load_solver

M = load_solver()
ROOT = Path(__file__).resolve().parents[2]


def completion(code, generated=7, prompt=100, cached=20):
    return ChatCompletion(
        id="test", created=0, model="test", object="chat.completion",
        choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "tool", "type": "function", "function": {
                    "name": "python", "arguments": json.dumps({"code": code}),
                },
            }],
        }}],
        usage={"completion_tokens": generated, "prompt_tokens": prompt,
               "total_tokens": generated + prompt,
               "prompt_tokens_details": {"cached_tokens": cached}},
    )


def game():
    board = np.zeros((64, 64), dtype=np.int16)
    result = SimpleNamespace(
        game_id="ls20", current_state=SimpleNamespace(
            frame=SimpleNamespace(data=board),
            raw=SimpleNamespace(state=M.arcengine.GameState.NOT_FINISHED, levels_completed=0),
            available_actions=[M.arcengine.GameAction.ACTION1, M.arcengine.GameAction.ACTION2],
        ),
        game_run=SimpleNamespace(state="playing", final_score=None), costs=[],
    )

    def execute(action, **usage):
        result.costs.append(usage)
        board[0, 0] += 1

    def finish(**usage):
        result.final_cost = usage
        result.game_run.final_score = 0
        result.game_run.state = "gave_up"

    result.execute_action = execute
    result.finish_game = finish
    return result


async def fake_client(agent, responses):
    await agent.client.close()
    replies = iter(responses)
    requests = []

    async def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return next(replies)

    async def close():
        pass

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
    agent.tokenizer_available = False
    return requests


@pytest.mark.asyncio
async def test_usage_includes_preflight_inspection_moves_and_final_stall(monkeypatch, capsys):
    monkeypatch.setenv("MY_AGENT3_LOG_DIR", "")
    monkeypatch.setenv("MY_AGENT3_TOOL_CALLS_PER_TURN", "2")
    monkeypatch.setenv("MY_AGENT3_MAX_NO_ACTION_RETRIES", "0")
    g = game()
    original_agent = M.PythonToolAgent
    responses = [completion("print('preflight-ok')", 3), completion("print('inspect')", 5),
                 completion("action(['ACTION1'])", 7), completion("print('inspect')", 9),
                 completion("print('inspect')", 11)]
    agents = []

    def factory(runtime, analyzer_timeout):
        agent = original_agent(runtime, analyzer_timeout)
        agents.append(agent)
        return agent

    monkeypatch.setattr(M, "PythonToolAgent", factory)
    solver = M.MyAgent3Solver(max_actions_per_game=22)
    # Set up the real clients before the solver's first request, with no network.
    original_preflight = original_agent.preflight

    async def preflight(agent):
        await fake_client(agent, responses)
        return await original_preflight(agent)

    monkeypatch.setattr(original_agent, "preflight", preflight)
    await solver._run_games([g])
    assert g.costs == [{"generated_tokens": 15, "uncached_input_tokens": 240}]
    assert g.final_cost == {"generated_tokens": 20, "uncached_input_tokens": 160}
    assert agents[0].runtime.generated_tokens == 35
    assert "inspection_retry_exhausted" in solver._model_failures[0]["error"]
    output = capsys.readouterr().out
    assert '"reason": "inspection_retry_exhausted"' in output
    assert '"generated_tokens": 35' in output


def test_failed_action_keeps_costs_and_batch_does_not_count_them_twice(monkeypatch):
    monkeypatch.setenv("MY_AGENT3_MAX_BATCH_ACTIONS", "2")
    g = game()
    runtime = M.PythonToolRuntime(g, 22, MemoryLogger())
    runtime.record_usage(completion("", 13, 100, 30).usage)
    execute = g.execute_action
    g.execute_action = Mock(side_effect=ValueError("engine rejected move"))
    assert "engine rejected move" in runtime.execute("action(['ACTION1'])")
    assert runtime.pending_generated_tokens == 13
    g.execute_action = execute
    runtime.action(["ACTION1", "ACTION2"])
    assert g.costs == [{"generated_tokens": 13, "uncached_input_tokens": 70},
                       {"generated_tokens": 0, "uncached_input_tokens": 0}]
    assert runtime.pending_generated_tokens == 0


def test_missing_usage_is_visible_and_cached_tokens_never_make_input_negative():
    runtime = M.PythonToolRuntime(game(), 22, MemoryLogger())
    runtime.record_usage(None)
    runtime.record_usage(completion("", 7, 10, 30).usage)
    assert runtime.responses_without_usage == 1
    assert runtime.uncached_input_tokens == 0
    assert ("model_usage_missing", {}) in runtime.logger.events


@pytest.mark.asyncio
async def test_same_board_retry_retains_inspection_and_adds_action_guidance():
    runtime = M.PythonToolRuntime(game(), 22, MemoryLogger())
    agent = M.PythonToolAgent(runtime)
    try:
        requests = await fake_client(agent, [completion("print('observed')"), completion("action(['ACTION1'])")])
        agent.max_tool_calls = 1
        agent.max_messages = 5
        assert not await agent.play_turn()
        inspected = list(agent.messages)
        await agent.retain_after_inspection()
        assert await agent.play_turn()
        request_messages = requests[1]["messages"]
        assert all(message in request_messages for message in inspected)
        assert sum(message["role"] == "user" and isinstance(message["content"], list)
                   for message in request_messages) == 1
        assert {"role": "user", "content": M.INSPECTION_RECOVERY_PROMPT} in request_messages
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_inspection_exhaustion_is_bounded_and_rejects_offline_success(monkeypatch):
    monkeypatch.setenv("MY_AGENT3_MAX_NO_ACTION_RETRIES", "2")
    solver = M.MyAgent3Solver()
    solver._preflight_ok = True
    solver._preflight_lock = asyncio.Lock()
    runtime = SimpleNamespace(game=game(), actions_taken=4, last_error=None,
                              terminal=lambda: False, game_over=lambda: False)
    calls = []
    recoveries = []

    async def turn():
        calls.append(1)
        return False

    async def retain():
        recoveries.append(1)

    reason = await solver._play_until_done(runtime, SimpleNamespace(
        play_turn=turn, retain_after_inspection=retain), MemoryLogger())
    assert reason == "inspection_retry_exhausted"
    assert len(calls) == 3 and len(recoveries) == 2
    assert solver._model_failures


def test_teardown_retries_endpoint_capture_and_clears_stale_error():
    module = ModuleType("teardown")
    module.PROCESS_POLL_SECONDS = 0.1
    attempts = []

    def capture(metrics, models, result):
        attempts.append(module.ENDPOINT_TIMEOUT_SECONDS)
        if len(attempts) == 1:
            result["final_metrics_error"] = "slow metrics"
        else:
            result.update(final_metrics_sha256="metrics", final_models_sha256="models")

    module.capture_endpoints = capture
    cleanup.configure(module)
    result = {}
    module.capture_endpoints(None, None, result)
    assert attempts == [3.0, 3.0]
    assert result["endpoint_capture_attempts"] == 2
    assert "final_metrics_error" not in result
    assert module.MAX_EXPLICIT_WAIT_SECONDS < 60


def test_teardown_rejects_changed_bundle_before_running_code(tmp_path):
    path = tmp_path / "serving_teardown.py"
    path.write_text("raise AssertionError('unreviewed code executed')")
    with pytest.raises(RuntimeError, match="Unreviewed serving teardown"):
        cleanup.run(path)


def test_teardown_terminal_failure_is_reported_and_still_raises(tmp_path, monkeypatch, capsys):
    path = tmp_path / "serving_teardown.py"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"port_closed": False, "final_metrics_error": "timeout"}))
    path.write_text('''from pathlib import Path
PROCESS_POLL_SECONDS = 0.1
def capture_endpoints(*args): pass
def validate_working_dir(): return Path(__file__).parent
def working_paths(root): return {"result": root / "result.json"}
def main(): raise RuntimeError("bounded terminal gate")
''')
    monkeypatch.setattr(cleanup, "PINNED_TEARDOWN_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="bounded terminal gate"):
        cleanup.run(path)
    output = capsys.readouterr().out
    assert "VLLM_TEARDOWN_FAILURE" in output
    assert '"port_closed": false' in output
    assert '"metrics_error": "timeout"' in output


def benchmark_execution():
    document = json.loads((ROOT / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb").read_text())
    tree = ast.parse("".join(document["cells"][16]["source"]))
    index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Try))
    return compile(ast.Module(body=tree.body[index:], type_ignores=[]), "<benchmark>", "exec",
                   flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.asyncio
@pytest.mark.parametrize("gameplay_failed", [False, True])
async def test_cleanup_failure_blocks_success_and_preserves_gameplay_error(tmp_path, capsys, gameplay_failed):
    (tmp_path / "teardown_commands.json").write_text(json.dumps(["echo cleanup-evidence; exit 7", "echo cleanup-second"]))

    async def run(**kwargs):
        pass

    ns = {"bm": SimpleNamespace(run=run, _save_json=Mock(), solver=SimpleNamespace(
        _model_failures=[{"error": "model failure"}] if gameplay_failed else [])),
        "soft_end": None, "target": None, "TRUE_SUBMISSION": True, "WORKING_DIR": tmp_path,
        "BUNDLE_DIR": tmp_path, "json": json, "sys": sys, "subprocess": subprocess,
        "_command_env": lambda: {}, "vllm_watchdog": SimpleNamespace(stop_background=Mock())}
    with pytest.raises(RuntimeError, match="Model failures" if gameplay_failed else "Cleanup failed") as error:
        await eval(benchmark_execution(), ns)
    if gameplay_failed:
        assert "Cleanup failed" in error.value.__notes__[0]
    output = capsys.readouterr().out
    assert "cleanup-evidence" in output and "cleanup-second" in output
    assert "BENCHMARK_COMPLETE" not in output
    assert not (tmp_path / "submission.parquet").exists()
    assert not (tmp_path / "offline_validation.json").exists()


@pytest.mark.asyncio
async def test_watchdog_error_and_command_timeout_still_run_remaining_cleanup(tmp_path, capsys):
    (tmp_path / "teardown_commands.json").write_text(json.dumps(["first", "second"]))

    async def run(**kwargs):
        pass

    runner = Mock(side_effect=[subprocess.TimeoutExpired("first", 60),
                               subprocess.CompletedProcess("second", 0, "second-finished", "")])
    ns = {"bm": SimpleNamespace(run=run, _save_json=Mock(), solver=SimpleNamespace(_model_failures=[])),
          "soft_end": None, "target": None, "TRUE_SUBMISSION": True, "WORKING_DIR": tmp_path,
          "BUNDLE_DIR": tmp_path, "json": json, "sys": sys,
          "subprocess": SimpleNamespace(run=runner), "_command_env": lambda: {},
          "vllm_watchdog": SimpleNamespace(stop_background=Mock(side_effect=RuntimeError("watchdog failed")))}
    with pytest.raises(RuntimeError, match="Cleanup failed") as error:
        await eval(benchmark_execution(), ns)
    assert "watchdog failed" in str(error.value)
    assert "TimeoutExpired" in str(error.value)
    assert runner.call_count == 2
    assert "second-finished" in capsys.readouterr().out


def test_embedded_teardown_matches_reviewed_wrapper():
    document = json.loads((ROOT / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb").read_text())
    tree = ast.parse("".join(document["cells"][13]["source"]))
    source = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "EMBEDDED_TEARDOWN_SOURCE" for t in node.targets))
    assert ast.literal_eval(source) == (ROOT / "scripts/duck_serving_teardown.py").read_text()
