# Submission validation and logging fix

The cancelled run never demonstrated completion or a valid Phase A artifact. Model HTTP errors were swallowed into gave_up game states, and the notebook originally wrote its placeholder before validating coverage and scores.

The solver now records model/preflight failures and unexpected solver exceptions independently of game score. After bm.run, the notebook rejects these failures. Offline validation requires the expected games in order, won/gave_up terminal states, final scores, no solver exception notes, and some recorded actions. Cancelled, crashed, missing, and unfinished runs fail.

Startup removes stale offline submission.parquet and offline_validation.json files. Only after gameplay coverage and the frozen scorer pass does the notebook create its Phase A placeholder. It reads the parquet back and checks the data, then writes offline_validation.json with game IDs, action count, solver name, and artifact paths. A completed execution with score zero can pass; this checks technical execution, not competitive quality. Real reruns retain the live gateway flow and never write the offline placeholder.

Development JSONL files contain full model/tool records, while notebook mirroring and board dumps default off; competition file logging defaults off. The existing watchdog stop and teardown finally block remains active on failure.

Validation: 12 regressions cover terminal coverage, zero-score completion, cancelled/crashed runs, missing scores/actions, exception notes, model-failure recording, stale artifact removal, and cleanup when model failures prevent placeholder creation. Notebook cells compile and the embedded source matches. TAAF, the frozen scorer, pandas/parquet, and the actual gateway are not available locally; their complete path must be checked on Kaggle.
