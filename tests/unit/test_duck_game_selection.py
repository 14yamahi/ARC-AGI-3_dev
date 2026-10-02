import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def definitions():
    path = Path(__file__).resolve().parents[2] / 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb'
    notebook = json.loads(path.read_text())
    tree = ast.parse(''.join(notebook['cells'][16]['source']))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_select_offline_game_ids'
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'PUBLIC_GAME_IDS' for target in node.targets)]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, tree


def test_ls20_prefix_and_exact_id_select_only_one_game():
    ns, _ = definitions()
    for selection in ('ls20', 'ls20-9607627b'):
        assert ns['_select_offline_game_ids'](ns['PUBLIC_GAME_IDS'], selection) == ['ls20-9607627b']


def test_all_and_multiple_game_selection():
    ns, _ = definitions()
    assert ns['_select_offline_game_ids'](ns['PUBLIC_GAME_IDS'], 'all') == list(ns['PUBLIC_GAME_IDS'])
    assert ns['_select_offline_game_ids'](ns['PUBLIC_GAME_IDS'], 'ls20,cn04') == ['ls20-9607627b', 'cn04-2fe56bfb']


@pytest.mark.parametrize('selection', ['', 'missing', 'ls20,ls20', 'ls20,'])
def test_invalid_selection_is_not_silently_replaced_by_all(selection):
    ns, _ = definitions()
    with pytest.raises(RuntimeError):
        ns['_select_offline_game_ids'](ns['PUBLIC_GAME_IDS'], selection)


def test_ambiguous_prefix_is_rejected():
    ns, _ = definitions()
    with pytest.raises(RuntimeError, match='exactly one'):
        ns['_select_offline_game_ids'](['ls20-a', 'ls20-b'], 'ls20')


def test_competition_selection_ignores_offline_filter(monkeypatch):
    ns, tree = definitions()
    monkeypatch.setenv('MY_AGENT3_OFFLINE_GAMES', 'ls20')
    games = [object(), object()]
    ns.update(TRUE_SUBMISSION=True, os=os, bm=SimpleNamespace(),
              _wait_for_gateway=Mock(), _competition_games=lambda: games,
              _select_offline_game_ids=Mock(side_effect=AssertionError('offline filter was used')))
    node = next(node for node in tree.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == 'TRUE_SUBMISSION')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<selection>', 'exec'), ns)
    assert ns['bm'].games is games
    ns['_select_offline_game_ids'].assert_not_called()
