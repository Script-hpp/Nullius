# Discovery: false horizon rejections (2026-10-12)

The deployed ETTh1 search started at 1791841533 and ended at 1791841621.
It examined SARAF (2606.04135), Time-VLM (2502.04395), and Data Embeddings
(2505.20716), rejecting all three with `Horizont nicht 96` and selecting no claims.
Evidence: the server's `/api/discover/replay`, retrieved after the user screenshot.

That message did not establish an incompatible paper result: the old condition
combined a missing/invalid/wrong model horizon field and a quote without `96`.
The replay does not store the extracted claim, so it cannot distinguish which
branch occurred in this production search. SARAF's source Table 4 contains an
ETTh1/horizon-96 SARAF MSE of 0.371; RAFT's value is 0.366.

An offline regression with that actual source table reproduces false rejection
when the model quotes only the result cells, or omits the structured horizon.
`tests/fixtures/saraf_main_results.tex` is that public table from
https://arxiv.org/abs/2606.04135, without executing any author code.

## Fix

- An aligned LaTeX row with verified dataset, method, metric, and target now
  supplies its horizon and full source row. This can recover a shortened quote
  or missing model horizon without guessing. Explicit conflicting model horizons
  remain blocked, as do other-method numbers, other horizons, and averages.
- Unverified horizon extraction is described as an extraction limit, with
  observed horizon/source horizon/quote evidence in the event. It is not cached
  as evidence against the paper. Historic generic horizon messages receive the
  same cautious UI explanation; stored events themselves remain unchanged.
- Discovery cache v5 re-examines previous v4 rejections on the next live search.
  Accepted older candidates remain importable and subject to strict run checks.

## Validation and limits

199 focused backend tests and 139 frontend tests passed. Two previously known
Discovery query-limit tests remain deselected (tests expect 3; production uses 5,
unchanged). Production build passed; lint has existing warnings, no errors.
Desktop/mobile production browser checks replayed this search through mocked
live endpoints: three extraction explanations, no false incompatible-horizon
assertion, no overflow or page errors. No paid search or experiment was started.

This does not prove the precise model output of the production incident, nor
guarantee selection in a fresh search. A model can still choose an unsupported
or averaged row. Horizon recovery is limited to unambiguous aligned LaTeX rows;
unknown layouts remain unverified. Retrieval and target guardrails stay active.
Deploy the new commit before another live ETTh1 search. Existing recorded searches
are not rewritten. No new environment variables or limit changes are required.
