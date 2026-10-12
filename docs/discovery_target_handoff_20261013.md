# Discovery -> reproduction target verification (2026-10-13)

The user started the correctly discovered SARAF 0.371 claim. The run stopped
before any experiment because role 2 generated this non-verbatim excerpt:

```text
ETTh1 & 96 & \underline{0.371} & \underline{0.399}
```

The original LaTeX table wraps ETTh1 in `multicolumn`, `multirow` and `rotatebox`.
The fabricated plain row fails exact source matching. The check's rejection was
correct; requiring another model to reconstruct a quote after Discovery had
already parsed the table was an avoidable integration gap. Previous source tests
covered Discovery through persistence and eligibility, not this subsequent gate
with a real SARAF source table.

## Fix

`claims_source` now uses the shared main-table selector on the **freshly fetched**
LaTeX paper source for explicit MSE/MAE at prediction length 96. It verifies the
requested method, original source quote, number, dataset, horizon and metric
column before producing `TargetEvidence` with the current source hash, URL and
retrieval metadata. It does not trust the Discovery quote as proof and needs no
additional numerical model call for an unambiguous source table.

The source selector now supports both MSE and MAE. Accuracy, unknown layouts,
ambiguous tables and unavailable primary sources retain the existing paths and
guards. The strict configured-target comparison stays unchanged: old SARAF 0.366
is still stopped when the independent source check reads SARAF 0.371, and a
changed fresh source value is blocked. A synthesized quote remains invalid.

Fresh deterministic evidence takes precedence over a malformed derived target
cache. Production `mode=live` does not rewrite that cache or historical claims,
measurements, verdicts or recordings. The stopped UI explains the failed check
instead of attributing a target-verification stop to sandbox/model instability.

## Validation

- `test_discovery_target_handoff.py` executes actual Discovery -> catalog adapter
  -> `Orchestrator.resolve_targets`, using separate fresh source evidence and
  forbidding a numerical model call. SARAF 0.371 and GLinear 0.382 both pass the
  subsequent strict target gate with exact source quotes and matching source hashes.
- Wrong old target, changed fresh source, the exact invalid user-reported quote,
  MAE extraction and a malformed derived cache have regression coverage.
- 335 focused backend tests passed; two previously known query-limit tests remain
  deselected (tests expect 3, production uses 5, unchanged). 140 frontend tests,
  production build and lint passed without errors; existing warnings remain.
- Four production browser cases: desktop/mobile target/method stops, correct
  stopped explanation, no false paper verdict/certificate, saved reason retained,
  no page errors. API endpoints were mocked; no paid run was started.
- Full Docker image built successfully, including Python import verification.
  The actual full SARAF source independently yielded MSE 0.371 and MAE 0.399,
  with verified columns and literal source quotes, without a model call.

After the main push, user redeploys and can retry the **existing 0.371 claim**;
another Discovery search is unnecessary. Budget/counter limits remain intact.
This resolves the source-quote stop, not a guarantee that the generated method
will complete on CPU or reach the paper's reported metric.
