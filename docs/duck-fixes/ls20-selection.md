# Offline ls20 selection fix

The run cell rebuilt the full public set after the customization hook, overriding an earlier bm.games restriction. Its final audit also assumed exactly 25 games.

The run cell now applies MY_AGENT3_OFFLINE_GAMES after offline discovery. It defaults to ls20; set it to all for the pinned public-25 run, or a comma-separated list such as ls20,cn04. Full IDs are also accepted. Unknown, ambiguous, empty, and duplicate selections fail explicitly. The final coverage check and audit count use the selected IDs. Full public dataset identity checks still protect against changed input environments.

Competition reruns ignore this filter and discover all live gateway games. The default single-game run is a development check, not a claim of full public performance.

Validation: eight regressions cover prefix/exact ID selection, all/multiple games, rejected selections, and the live competition branch ignoring the offline filter. Notebook cells compile and the embedded source matches. Use this branch to isolate selection changes; the combined branch fixes the solver failures too.
