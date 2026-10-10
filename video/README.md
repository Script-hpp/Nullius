# Nullius film v9

Local capture and rendering tools. The English narration, timing and scope come from [video_script.md](../docs/video_script.md). Production measurements are loaded from the website exports, not entered independently in the film. Generated assets are kept in ignored `out/`.

Requirements: Node 22, Python 3, Chromium for Playwright, FFmpeg, `espeak-ng`, and Poppler (`pdftocairo`, for the optional outlined logo export).

Start the current app in another terminal from `web/` with `npm run dev -- --host 127.0.0.1 --port 5186`. Then, from the repository root:

```bash
npm ci --prefix video
# Install Chromium if it is not already available:
cd video
npx playwright-core install chromium
cd ..

python video/compose.py
node video/capture.mjs http://127.0.0.1:5186
python video/narrate.py
node video/render.mjs

ffmpeg -y -i video/out/picture.mp4 -i video/out/narration.wav -i video/out/film-v9.en.srt \
  -map 0:v -map 1:a -map 2:s -c:v copy -c:a aac -b:a 160k -c:s mov_text \
  -metadata:s:s:0 language=eng -movflags +faststart video/out/film-v9.mp4
```

Capture blocks `/api/` requests: it navigates the real local product, opens the actual certificate popup and changes the What-if slider, using stored evidence. It does not launch experiments. `capture-check.json` records browser errors and the captured run. The narration is synthesized locally; listen to the complete film before publishing or replacing it with a human recording.

`node video/preview.mjs` exports representative frames. `node video/logo.mjs` outlines the existing product wordmark through Chromium PDF and Cairo so the GitHub SVG needs no installed fonts or external resources. It does not redesign the mark.

The MP4 is a local review artifact, not an uploaded video. The proposed demo hostname remains unpublished. The next gate is the [human product test](../docs/product_test_20261010.md).
