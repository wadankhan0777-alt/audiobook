# CLAUDE.md

AI audiobook studio: turns public-domain / user-owned books into multi-voice audiobooks with emotion, ambience and synced SFX.

Full design: `docs/ARCHITECTURE.md`. Read only the section you need.

## Status
Architecture drafted; owner decisions: English only, rented GPU OK, personal use only (see ARCHITECTURE.md §9). **Do not write app code until the owner explicitly approves starting a phase.**

## Stack
- `web/` Next.js (TypeScript, App Router). UI only, no business logic.
- `api/` Python 3.12, FastAPI, SQLAlchemy + Alembic, SQLite (WAL). Postgres-ready, don't use SQLite-only features.
- `worker/` single GPU worker loop that claims rows from the `jobs` table. No Redis/Celery.
- `engines/` adapters: `tts/` (qwen3, moss, kokoro), `sfx/` (moss_sfx), `align/` (ctc_forced_aligner). App code only talks to the `TTSEngine` / `SFXEngine` / `Aligner` interfaces.
- FFmpeg via subprocess for all mixing and export.
- Data on disk: `/data/projects/<project_id>/{segments,chapters,exports}`.

## Permanent rules
- Work in small phases from the roadmap; one phase per branch/PR; owner approves each phase.
- Never add a paid API. Local models only. Never add a dependency without checking code AND weights licence.
- Every claim about a model feature must be verified in its repo/code first.
- Gutenberg: never scrape gutenberg.org pages. Use the catalog files / Gutendex / mirror, ≥2 s between file downloads.
- Open Library: metadata only. Never fetch or convert borrowed/DRM books.
- Segments are the unit of work: every generated audio file maps to one `segments` row with status + version, so generation is always resumable and any line can be regenerated alone.
- Voice profiles are locked once approved; never silently change a character's voice mid-book.
- Worker loads one model family at a time and unloads it after.
- Store licence + source for every SFX asset.
- Don't duplicate functionality: check `engines/` and existing services before adding new ones.
- Keep explanations short. Don't re-read unchanged files.

## Hardware
No local GPU. Model work runs on a rented cloud GPU (24 GB target). Everything else must also run CPU-only (Kokoro previews, parsing, mixing, UI/API).
