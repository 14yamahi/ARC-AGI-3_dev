# Python tool compatibility fix

The run repeatedly called ACTION6 without coordinates, used the prompt's unsupported MOUSE name, and tried builtins or imports excluded from the Python tool.

MOUSE now resolves to ACTION6. Every click must supply both coordinates, as x/y or row/col. The parser rejects missing coordinates, booleans, fractional coordinates, and out-of-board clicks before entering the game engine. The tool now supplies type, isinstance, hasattr, ordinary exception classes, numpy/np, hashlib, and last_error. Existing instruction, code-size, output-size, and direct-ASCII limits remain enforced.

The prompt explains coordinate-bearing ACTION6 calls, action-name-only predictions, compact result['change'] versus full last_action_result['change_summary'], TransitionView attributes, and splitting crop strings into rows. This removes mismatches shown in the saved tool errors without choosing game actions for the model.

Validation: nine regressions cover a real engine-input object with click coordinates, MOUSE normalization, malformed click rejection, builtins/imports, and documented result schemas. Notebook cells compile and the embedded source matches. Model reasoning and game scores still need measurement.
