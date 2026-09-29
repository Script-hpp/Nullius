# Evidence view

Static React app (Vite, TypeScript, Tailwind, Recharts, React Flow, Framer Motion) that replays real runs of the reproduction agent:
what Tavily found (left) next to what the sandbox measured (right), the branch tree, and a step-by-step replay from `events.jsonl`.

No backend, no API keys: the run data are JSON files in `public/data/`, so any static web server works (nginx, GitHub Pages).

```bash
# 1. export runs (only runs that may be published; never prototype runs on free endpoints)
uv run python scripts/export_site_data.py runs/<paper>/<timestamp> [more runs]
# 2. build and serve
cd web && npm install && npm run build       # output in web/dist
npx vite preview --port 4173                  # local check
```

`base` is `./`, so `dist/` can be served from any sub-path. Runs without `events.jsonl` (older runs) are replayed from the dossier.
