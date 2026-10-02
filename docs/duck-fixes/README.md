# Duck notebook fix branches

The common baseline is fix/duck-review-baseline (12630d8), containing the October 2 cancelled notebook and a test loader that replaces only the unavailable TAAF imports. Each problem branch starts there and can be reviewed independently:

| Branch | Change | Tests |
| --- | --- | ---: |
| fix/duck-context-budget | Server token counting, complete context groups, bounded HTTP 400 recovery | 4 |
| fix/duck-tool-interface | Coordinate click validation, MOUSE alias, builtins and prompt/API alignment | 9 |
| fix/duck-async-concurrency | Async model requests, bounded games, preflight lock and deadline cleanup | 3 |
| fix/duck-ls20-selection | Selection after discovery; audits match selected games | 8 |
| fix/duck-submission-validation | Explicit failures, delayed placeholder, artifact check and quieter logs | 12 |

fix/duck-all-fixes contains all changes. It additionally makes tokenization asynchronous, awaits context eviction/inspection retention, records model failures in the async play loop, and validates the selected offline IDs. Its 39 regressions include three notebook checks for syntax/output freshness, embedded source identity, and ls20 selection connected to the final audit. Ruff checks pass for changed Python files.

Every fix notebook embeds its checkout's my_agent3.py and prints a SHA-256 identity. This makes local fixes effective on Kaggle without updating the agent dataset. The model/runtime/source-bundle attachments from the successful setup are still required. Regenerate after solver edits with `python scripts/embed_duck_agent.py`; this clears saved execution output. The baseline retains the cancelled run evidence.

## Next Kaggle check

1. Upload the combined branch's duck-qwen3-8-flash-next-nvfp4-mtp.ipynb. Keep the same working input attachments and GPU selection as the reviewed run.
2. Use a fresh session. Offline selection defaults to ls20. Expect OFFLINE_SELECTION games=1 with ls20-9607627b, an embedded agent SHA-256, BENCHMARK_COMPLETE, and OFFLINE_AUDIT runs=1. Inspect my_agent3_logs for model errors, tokenizer fallback, or repeated tool errors.
3. Confirm score.json, offline_validation.json, and submission.parquet exist and the Save & Run completes. A zero score proves only execution.
4. Set `os.environ["MY_AGENT3_OFFLINE_GAMES"] = "all"` before the run cell for full public validation. Check coverage, errors, runtime, and scores before a competition submission. Concurrency can be adjusted using MY_AGENT3_CONCURRENCY before the settings cell; no speedup is certified yet.
5. Competition reruns always use the live gateway's full game set, irrespective of the offline filter.

Local tests use production methods with mocked model responses, games, and TAAF imports. They do not run the 135 GB checkpoint, the mounted vLLM build, the frozen scorer, pandas/parquet, or the live gateway. No notebook was uploaded or submitted by this work.

## Existing checkouts

The branch worktrees are under /tmp/duck-fixes-20261002/<problem>, with the combined checkout at /tmp/duck-fixes-20261002/all-fixes. Use these checkouts to inspect or upload the notebook. The original main checkout and its notebook moves were preserved; avoid switching it over untracked notebooks. The committed branches remain available if temporary checkouts are later removed.
