# Follow-up to the October 2 ls20 run

The executed notebook is preserved in `run-evidence/ls20-20261002-before-runtime-fixes.ipynb`.
It completed ls20 with five recorded actions, zero levels, zero recorded tokens,
and a serving teardown exception. Its per-game JSONL and teardown JSON were not
provided, so the exact original stop reason and failed cleanup gate remain unknown.

## Cleanup

The notebook embeds `scripts/duck_serving_teardown.py` alongside the solver.
The wrapper loads the attached bundle's teardown only after verifying its SHA-256
(`c48368e330abf2574b155b42041bd5d42ea556d53340ccbbc5c8788acaa0eb19`).
This is the public smoke-v1 bundle's reviewed teardown, downloaded for inspection.
The wrapper keeps its process-ownership checks and full shutdown gate. Metrics
and model endpoint capture now allow three seconds per request and two attempts;
TERM/KILL/port/GPU-query waits allow 8/4/2/5 seconds. The explicit wait bound is
31.3 seconds, within the notebook's 60-second subprocess limit.

Cleanup stdout and stderr are captured and forwarded to notebook output. Failed
cleanup reports the exact gate evidence from `vllm-server-teardown.json`. Every
teardown command is attempted even after a watchdog or command failure. Cleanup
failures raise; when gameplay already failed, cleanup errors become exception
notes on the original error. `BENCHMARK_COMPLETE`, the offline placeholder, and
`offline_validation.json` are emitted only after cleanup succeeds. The offline
audit records `cleanup_ok: true`.

These waits address transient endpoint latency and delayed release; they do not
certify which gate failed in the saved run. A remaining identity conflict,
surviving process, GPU query failure, or missing metrics still fails the run.

## Early give-up

Inspection retries previously appended a duplicate board state. Context eviction
could then discard the inspection exchanges the retry needed. The agent now
adds a state only when its action count changes. Recovery instructions explicitly
ask for a legal discriminating action and explain prediction/replan requirements;
eviction recognizes this instruction as feedback rather than a new board state.
Recovery remains bounded (four retries by default, configurable with
`MY_AGENT3_MAX_NO_ACTION_RETRIES`, including zero). Exhaustion is recorded as a
solver failure, so it cannot pass an offline execution audit. Every game prints
`MYAGENT3_STOP` with its stop reason, action budget, token costs, and missing-usage
response count, including on timeout/cancellation. The action cap remains 22.

## Token accounting

Successful preflight, inspection, and gameplay responses accumulate completion
tokens and prompt tokens minus cached prompt tokens. Costs are passed through
TAAF's `execute_action(generated_tokens=..., uncached_input_tokens=...)` on the
next successful action. Failed actions retain the costs. Subsequent actions in
one batch receive zero until another model response arrives. Any final no-action
cost is passed through `finish_game(...)`, including stalls/timeouts/cancellation.
Missing server usage is reported explicitly rather than estimated.

The API was checked against [TAAF's source](https://github.com/Tufalabs/duck-harness/blob/main/tufa-arc-agi-framework/src/taaf/game.py).
The regenerated notebook embeds both reviewed local sources, clears old outputs
and Papermill execution metadata, and retains the pinned serving configuration.

## Validation and next run

Local regressions exercise actual model-loop/runtime methods with mocked model
responses, costs across actions and final turns, failed and batched actions,
retry context retention and exhaustion, endpoint retries, teardown rejection,
terminal failure evidence, and notebook error/artifact ordering. They also
check both embedded source identities.

All 51 targeted regressions and Ruff checks pass. A separate compatibility probe
used the real downloaded TAAF `Game` implementation: 13 generated tokens on an
action plus seven final no-action tokens produced `tokens=20` in its finish output
and the same totals in serialized run data. The pinned teardown's original
ownership self-tests also pass. These probes used no model inference or GPU.

On a fresh Kaggle run, inspect `MYAGENT3_STOP`, `VLLM_SERVER_TEARDOWN` (or
`VLLM_TEARDOWN_FAILURE`), `TEARDOWN_COMPLETE`, and `OFFLINE_AUDIT`. If it fails,
download `my_agent3_logs`, `vllm-server-teardown.json`, and the server log.
Local tests cannot certify GPU release on Kaggle or improved puzzle scores.
