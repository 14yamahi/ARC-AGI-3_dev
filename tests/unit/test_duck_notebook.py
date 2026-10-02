import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace


def notebook():
    root = Path(__file__).resolve().parents[2]
    return root, json.loads((root / 'duck-qwen3-8-flash-next-nvfp4-mtp.ipynb').read_text())


def test_all_cells_compile_and_have_no_stale_run_output():
    _, document = notebook()
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{index}', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            assert not cell['outputs']
            assert cell['execution_count'] is None


def test_embedded_solver_matches_reviewed_source():
    root, document = notebook()
    tree = ast.parse(''.join(document['cells'][13]['source']))
    source = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'EMBEDDED_AGENT_SOURCE' for target in node.targets))
    assert ast.literal_eval(source) == (root / 'agents/templates/my_agent3.py').read_text()


def test_actual_offline_branch_selects_ls20_and_audit_uses_that_selection(monkeypatch):
    _, document = notebook()
    tree = ast.parse(''.join(document['cells'][16]['source']))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {'_select_offline_game_ids', '_validate_offline_runs'}
                   or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                   and target.id == 'PUBLIC_GAME_IDS' for target in node.targets)]
    ns = {'TRUE_SUBMISSION': False, 'os': os, 'Path': Path, 'bm': SimpleNamespace()}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<definitions>', 'exec'), ns)
    games = [SimpleNamespace(env_name=game_id) for game_id in ns['PUBLIC_GAME_IDS']]
    ns['_offline_games'] = lambda path: games
    monkeypatch.setenv('MY_AGENT3_OFFLINE_GAMES', 'ls20')
    selection = next(node for node in tree.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == 'TRUE_SUBMISSION')
    exec(compile(ast.Module(body=[selection], type_ignores=[]), '<selection>', 'exec'), ns)
    assert [game.env_name for game in ns['bm'].games] == ['ls20-9607627b']
    validation_call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name) and node.func.id == '_validate_offline_runs')
    assert isinstance(validation_call.args[1], ast.Name)
    assert validation_call.args[1].id == 'OFFLINE_GAME_IDS'
    run = SimpleNamespace(game_id='ls20-9607627b', state='gave_up', final_score=0, history=[1])
    assert ns['_validate_offline_runs']([run], ns['OFFLINE_GAME_IDS'], []) == 1
