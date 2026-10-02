import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.unit.my_agent3_test_support import MemoryLogger, load_solver

M = load_solver()


@pytest.mark.asyncio
async def test_games_overlap_and_respect_concurrency_limit():
    solver = M.MyAgent3Solver(concurrency=2)
    active = 0
    maximum = 0
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = []
    async def play(game):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            entered.set()
        try:
            await release.wait()
        finally:
            active -= 1
            finished.append(game)
    solver._play_one = play
    task = asyncio.create_task(solver._run_games(list(range(5))))
    await asyncio.wait_for(entered.wait(), 1)
    assert maximum == 2
    release.set()
    await task
    assert sorted(finished) == list(range(5))
    assert maximum == 2


@pytest.mark.asyncio
async def test_cancellation_waits_for_active_children_and_does_not_start_queued_games():
    solver = M.MyAgent3Solver(concurrency=2)
    started = []
    closed = []
    ready = asyncio.Event()
    async def play(game):
        started.append(game)
        if len(started) == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(game)
    solver._play_one = play
    task = asyncio.create_task(solver._run_games(list(range(10))))
    await asyncio.wait_for(ready.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(closed) == sorted(started)
    # Cancellation must cancel queued waiters before releasing slots.
    assert len(started) == 2


@pytest.mark.asyncio
async def test_preflight_runs_once_and_timeouts_finalize_games_and_close_clients():
    preflights = []
    closed = []
    loggers = []
    class Agent:
        def __init__(self, runtime, analyzer_timeout):
            self.runtime = runtime
            assert analyzer_timeout == 7
        async def preflight(self):
            preflights.append(1)
            await asyncio.sleep(0)
            return True
        async def play_turn(self):
            await asyncio.Event().wait()
        async def close(self):
            closed.append(self.runtime.game)
    def logger(game):
        result = MemoryLogger()
        loggers.append(result)
        return result
    def runtime(game, cap, logger):
        return SimpleNamespace(game=game, actions_taken=0, last_error=None,
                               terminal=lambda: False, game_over=lambda: False)
    games = []
    for i in range(3):
        game = SimpleNamespace(game_run=SimpleNamespace(final_score=None))
        def finish(game=game):
            game.game_run.final_score = 0
        game.finish_game = finish
        games.append(game)
    solver = M.MyAgent3Solver(concurrency=3, analyzer_timeout=7, max_runtime_s_per_game=0.03)
    with patch.object(M, "RunLogger", logger), patch.object(M, "PythonToolRuntime", runtime), patch.object(M, "PythonToolAgent", Agent):
        await solver._run_games(games)
    assert len(preflights) == 1
    assert len(closed) == 3
    assert all(game.game_run.final_score == 0 for game in games)
    assert all(any(event == "solver_stopped" and data["reason"] == "runtime_budget_exhausted"
                   for event, data in logger.events) for logger in loggers)
