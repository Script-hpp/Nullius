# Benchmark v4: incidents

Rule (pre-registration `../benchmark_v2.md`, section on infrastructure aborts): an infrastructure abort is not an agent failure; the generation is restarted in full, half-finished runs write no rows.

## 2026-10-02 03:01 CEST: process 2 died on a sandbox connection error

- Process: `run_benchmark.py --v4 shchur_gat_2018 velickovic_gat_2018 velickovic_dgi_2019` (PID 782740), log `logs/bench_v4/p2.log` lines 563-719.
- Error: `httpx.ConnectError: All connection attempts failed`, then `contree_sdk.sdk.exceptions.api.ContreeTransportError` on `GET https://api.tokenfactory.nebius.com/sandboxes/v1/operations/...` (Nebius sandbox API unreachable). The exception was not caught, so the whole process ended; no "abgebrochen" row was written.
- Affected: `velickovic_gat_2018`, arm A or B generation in progress at that moment (no row). Store at that time: 37 runs, 5.94 $ gross.
- Action (03:2x CEST): API reachable again; remaining runs restarted with unchanged code (`git diff bf520c1 HEAD -- src scripts/run_benchmark.py` empty), as two processes split by paper so they cannot overlap: `--v4 velickovic_gat_2018` (log `p2b.log`) and `--v4 velickovic_dgi_2019` (log `p2c.log`). The script skips runs already in the store.

## Budget check across processes

`run_benchmark.py` reads the store once at start and then adds only its own rows, so a process does not see the spend of the others. The pre-registered stop (8 $ gross in the store) is therefore enforced from outside: the reviewer's watcher sums `cost_usd_run` per run in `runs/benchmark_v4.jsonl` every minute and stops the remaining processes at 8.00 $ gross. Any such stop is recorded here.
