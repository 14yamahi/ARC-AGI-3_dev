"""Load the production solver with only the unavailable TAAF package replaced."""
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace


@dataclass
class StubSolver:
    concurrency: int = 1
    max_runtime_s_per_game: float = 60.0
    analyzer_timeout: float = 120.0


def load_solver():
    path = Path(__file__).resolve().parents[2] / "agents/templates/my_agent3.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name == "taaf.game" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "taaf.solver"
    )]
    module = ModuleType("duck_solver_under_test")
    module.__file__ = str(path)
    module.taaf = SimpleNamespace(game=SimpleNamespace(Game=object))
    module.Solver = StubSolver
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class MemoryLogger:
    def __init__(self):
        self.events = []

    def record(self, event, **data):
        self.events.append((event, data))

    def board(self, event, frame):
        self.record(event, level=frame.level)

    def close(self):
        pass
