"""Embed this checkout's solver in the Kaggle notebook and clear stale run output."""
import hashlib
import json
import sys
from pathlib import Path


def embed(root):
    path = root / "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
    notebook = json.loads(path.read_text())
    cell = notebook["cells"][13]
    old_source = "".join(cell["source"])
    agent = (root / "agents/templates/my_agent3.py").read_text()
    digest = hashlib.sha256(agent.encode()).hexdigest()
    source = '''from pathlib import Path
import importlib
import os
import sys

# Embed this checkout's solver so attached datasets cannot select stale code.
EMBEDDED_AGENT_SOURCE = ''' + repr(agent) + '''
dev_dir = Path("/kaggle/working/my_agent")
dev_dir.mkdir(parents=True, exist_ok=True)
(dev_dir / "my_agent3.py").write_text(EMBEDDED_AGENT_SOURCE, encoding="utf-8")
if str(dev_dir) not in sys.path:
    sys.path.insert(0, str(dev_dir))
importlib.invalidate_caches()
sys.modules.pop("my_agent3", None)
my_agent3 = importlib.import_module("my_agent3")
bm.solver = my_agent3.MyAgent3Solver(max_actions_per_game=22)
print("Custom agent module:", my_agent3.__file__)
print("Custom agent sha256:", ''' + repr(digest) + ''')
print("Solver:", type(bm.solver).__name__)

'''
    if '"" if TRUE_SUBMISSION' in old_source:
        source += '''# Keep complete development traces in files without flooding notebook output.
os.environ["MY_AGENT3_LOG_DIR"] = "" if TRUE_SUBMISSION else "/kaggle/working/my_agent3_logs"
os.environ.setdefault("MY_AGENT3_LOG_BOARDS", "0")
os.environ.setdefault("MY_AGENT3_LOG_STDOUT", "0")
'''
    else:
        source += '''os.environ["MY_AGENT3_LOG_DIR"] = "/kaggle/working/my_agent3_logs"
os.environ["MY_AGENT3_LOG_BOARDS"] = "1"
os.environ["MY_AGENT3_LOG_STDOUT"] = "1"
'''
    cell["source"] = source.splitlines(keepends=True)
    if "The Duck prompts" in "".join(notebook["cells"][0]["source"]):
        notebook["cells"][0]["source"] = (
            "## About this fork\n\nBased on Tufa Labs’ Duck harness, with credit to Jeroen Cottaar and Tufa Labs.\n"
            "This notebook selects the embedded MyAgent3Solver. Serving uses the pinned\n"
            "Qwen3.8-Flash-Next-NVFP4 model, offline vLLM runtime, and three-token MTP.\n"
            "Solver scheduling and validation behavior are documented with the fix branch.\n"
        ).splitlines(keepends=True)
    for code_cell in notebook["cells"]:
        if code_cell["cell_type"] == "code":
            code_cell["execution_count"] = None
            code_cell["outputs"] = []
            code_cell.get("metadata", {}).pop("execution", None)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    return digest


if __name__ == "__main__":
    checkout = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    print(embed(checkout))
