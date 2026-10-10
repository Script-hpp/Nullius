# Film v9 — local review artifact

Artifact: `video/out/film-v9.mp4` in the CEO workspace, **150.000 seconds**, **1920 × 1080**, **24 fps**, H.264 video, AAC audio and English subtitle track. The rendered MP4 is ignored in Git; its sources and reproduction instructions are in `video/`.

Built from the current website at `e925298` and stored production exports. App views are labelled recorded. GLinear is the main evidence path, with actual candidate-2 repair errors followed by final-code seed measurements. TSMixer, SparseTSF and GLinear appear separately from the 7/8 in-sample development agreement and RLinear 0/1 held-out miss. The 0/48 planted-control count is explicitly v4, paper-only. The source-validation warning remains visible. PDF footage comes from the actual export popup; its saved PDF contains the numerical result and source limitations.

Checks completed:

- Local capture: no JavaScript page errors; `/api/` traffic blocked; no new experiments or searches.
- Representative frames and a contact sheet from the final MP4 inspected, including the repaired source view. Development-count typography also checked at 640 × 360.
- Final duration and stream metadata verified with `ffprobe`; the entire final MP4 decoded without errors using FFmpeg.
- Locally synthesized English speech fits every scene without truncation. Timed English subtitles are supplied as `video/out/film-v9.en.srt` as well as the MP4 track.
- Actual GLinear PDF opened and text extracted; preliminary status, seed spread and table-column warning remain in the saved document.

The final audio has **not been listened to by a person**. This is a review cut with local synthetic speech, not a published or audio-approved submission. Watch and listen before uploading. The separate human product test also remains pending in [product_test_20261010.md](product_test_20261010.md).
