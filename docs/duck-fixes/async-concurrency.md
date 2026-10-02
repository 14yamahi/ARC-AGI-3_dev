# Asynchronous game concurrency fix

The notebook configured concurrency=28, but MyAgent3Solver explicitly awaited games sequentially and used a synchronous OpenAI client.

The solver now uses AsyncOpenAI with an asynchronous HTTP client and a bounded semaphore for active games. It runs one shared preflight under a lock. Python tools and engine actions stay on the event-loop thread because their stdout redirection and shared action enums make whole-game thread workers unsafe. Model requests can overlap without running game actions in parallel threads. Cancellation cancels and awaits the active and queued tasks, finalizes active game runs, and closes clients and loggers.

The per-game runtime setting now bounds asynchronous play, and analyzer_timeout sets the HTTP read timeout unless MY_AGENT3_READ_TIMEOUT overrides it. A synchronous engine action cannot be interrupted by this async deadline while it is executing. The notebook defaults to eight active games to match the pinned vLLM sequence cap; set MY_AGENT3_CONCURRENCY before the settings cell to compare other limits.

Validation: three regressions cover overlapping games with a concurrency cap, cancellation without starting queued games, and single preflight plus deadline finalization/client cleanup. Notebook cells compile and the embedded source matches. Throughput, GPU utilization, and full-set runtime require the Kaggle model and environments; eight is a starting configuration, not a measured optimum.
