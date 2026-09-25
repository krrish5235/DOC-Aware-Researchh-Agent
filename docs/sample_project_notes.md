# SAMPLE DOC 1 — replace with your own notes

This file is sample content so the project works out of the box.
Delete it and add your own .md / .txt / .pdf files to docs/, then re-run `python ingest.py`.

# Aurora Dashboard — Project Notes

**What it is:** Aurora is an internal analytics dashboard for tracking weekly
experiment metrics across three product teams.

## Architecture decisions (2026-02-14 meeting)

- **Database:** PostgreSQL via SQLAlchemy, NOT MongoDB. Rationale: the metrics
  are strongly relational and we already run Postgres in prod, so a second
  database would double the ops burden.
- **Frontend:** React + Vite. Chose Vite over CRA because CRA is deprecated and
  Vite's dev server starts in under a second on our monorepo.
- **Charts:** Recharts, not D3, because Recharts is declarative and the team
  already knows React idioms. D3 was rejected as too low-level for our timeline.

## API endpoints

- `GET /api/metrics?team=<name>&week=<iso-week>` — returns the metric rows.
- `POST /api/experiments` — registers a new experiment; requires `team_id`.
- `GET /api/health` — used by the deploy check.

## Known issues

- The weekly rollup job (`jobs/rollup.py`) double-counts experiments whose
  `started_at` falls exactly on a Monday 00:00 UTC. Bug ticket AUR-142, open.
- Large CSV uploads (>50MB) time out behind the nginx default proxy timeout.

## Roadmap (next quarter)

1. Move the rollup job from cron to a Celery queue.
2. Add per-team auth scopes (currently everyone sees all teams).
3. Export-to-Sheets button for the weekly report.
