# AI Audiobook Studio: Technical Architecture (v0.1, draft for approval)

Status: **research + design only, nothing implemented.** Research date: 2026-11-26.

Legend for claims: **[V]** verified by inspecting the actual repo or source code, **[D]** taken from official docs/model cards, **[E]** my estimate, to be measured in Phase 0, **[?]** not yet verified (site was unreachable from the research container).

---

## 1. Key findings (read this first)

1. **No open model does "consistent cloned character voice + per-line emotion instruction" in one call.** [V]
   - Qwen3-TTS `generate_voice_clone()` has no `instruct` argument (checked `qwen_tts/inference/qwen3_tts_model.py:470`). Instructions only work on the VoiceDesign and CustomVoice (9 fixed preset speakers) models.
   - MOSS-TTS v1.5 cloning has no instruction field either (params: `text`, `language`, `reference`, `tokens`). MOSS-VoiceGenerator takes instructions but has no reference audio, so it can't keep one voice stable across a book.
   - **Design consequence:** each character gets an *emotion reference bank* (neutral, angry, fearful, sad, whisper, excited...) made once with a voice-design model, then every line is cloned from the bank entry matching its emotion. A speaker-similarity check rejects drifted references. See section 4.
2. **Every candidate engine is Apache-2.0 for both code and weights.** [V code / D weights]
3. **You will need a rented GPU.** You're phone-only and this cloud container has no GPU. Plan for an on-demand cloud GPU (24 GB card) to run the models. The app itself (UI + API) can run anywhere. [E]
4. **Open Library borrowing can't feed the TTS.** Borrowed books are DRM-protected (Adobe ID / BookReader) 14-day loans [D]. Converting them into audio would break the loan terms. Open Library is used for **metadata, covers, and finding public-domain full texts only**.
5. **Project Gutenberg bans scraping the website.** Automated access gets IP-blocked; the sanctioned routes are the rsync mirror and the nightly catalog files [D]. We use a self-hosted Gutendex (MIT [V]) built from those catalog files.
6. **Standard Ebooks is the best-quality source.** Every book has a public GitHub repo with clean per-chapter XHTML and Library of Congress subjects [V]. OPDS feeds / bulk zips need a Patrons Circle login [D].

---

## 2. TTS engine comparison

| | **Qwen3-TTS** | **MOSS-TTS family** | **Kokoro** |
|---|---|---|---|
| Repo | QwenLM/Qwen3-TTS | OpenMOSS/MOSS-TTS (+ MOSS-TTSD) | hexgrad/kokoro |
| Last commit [V] | 2026-03-17 | 2026-09-06 | 2025-08-06 |
| Code / weights licence | Apache-2.0 [V] / Apache-2.0 [D] | Apache-2.0 [V] / Apache-2.0 [D] | Apache-2.0 [V] / Apache-2.0 [D] |
| Sizes | 0.6B, 1.7B | TTS-v1.5 8B, Local-v1.5 4B, VoiceGenerator 1.7B, TTSD 8B, Nano 0.1B | 82M |
| Voice design from text | Yes, 1.7B-VoiceDesign [D] | Yes, MOSS-VoiceGenerator [D] | No (fixed voicepacks) |
| Voice cloning | Yes, Base model, 3 s ref, reusable `voice_clone_prompt` [V] | Yes, v1.5 "more stable cloning" [D] | No |
| Emotion via instruction | VoiceDesign + CustomVoice only, **not** on clones [V] | VoiceGenerator only, **not** on clones [V] | No |
| Long-form | per-call; no single-run length claim | up to 1 h single run (TTS, TTSD) [D] | per-sentence |
| Multi-speaker dialogue in one pass | No | Yes, MOSS-TTSD `[S1] [S2]` tags [D] | No |
| Pause / duration control | via text/instruct | explicit `[pause 1.2s]` + token-level duration (1 s ≈ 12.5 tokens) [D] | speed param |
| Languages | 10 incl. English [D] | 31 (v1.5) [D] | ~9 [D] |
| Output | 24 kHz [E] | 24 kHz (8B) / 48 kHz stereo (Local-v1.5) [D] | 24 kHz [D] |
| Serving | qwen-tts pip, vLLM-Omni (offline) [D] | Transformers 5, SGLang (~3x faster), vLLM-Omni, llama.cpp GGUF [D] | pip, CPU-friendly |
| VRAM [E] | 1.7B bf16 ≈ 3.5 GB weights, ~6–10 GB running | 8B bf16 ≈ 16 GB weights → 24 GB card; GGUF low-memory mode fits 8 GB [D] | <2 GB, runs on CPU |

### Recommendation

- **Primary engine: Qwen3-TTS 1.7B (VoiceDesign → Base clone).** The design-then-clone workflow is documented for exactly our use ("consistent character voice across many lines") [D], it's ~5x smaller than MOSS-TTS 8B, so it runs alongside the LLM and SFX model on one 24 GB GPU.
- **Premium / alternative engine: MOSS-TTS v1.5 (+ MOSS-VoiceGenerator for design).** Better pause/duration control and long-form claims, but 8B. Plug-in adapter, chosen per project.
- **Preview / CPU fallback: Kokoro 82M.** Fast drafts and UI previews without a GPU. Not for final output (no emotion, no custom voices).
- **Not used for now: MOSS-TTSD.** One-pass dialogue clashes with per-line regeneration (you can't regenerate one line of a 10-minute generation). Revisit later as an optional "scene mode".
- **The final choice between Qwen3-TTS and MOSS-TTS is decided by a blind listening bake-off in Phase 0** on the same chapter. Quality claims above are the vendors', not mine.

---

## 3. Book sources

| Source | Use | Access method | Licence / legal |
|---|---|---|---|
| **Standard Ebooks** | Primary (best formatting) | Clone per-book GitHub repo `standardebooks/<author>_<title>` (per-chapter XHTML, `content.opf` subjects) [V]; OPDS with Patrons Circle login optional [D] | Their work CC0; texts US public domain [V] |
| **Project Gutenberg** | Largest catalog (~75k) | Self-hosted **Gutendex** fed by nightly catalog files; EPUB downloads via mirror, ≥2 s between requests [D] | US public domain; strip PG licence header/footer, or keep it if the "Project Gutenberg" trademark is shown [D] |
| **Open Library** | Metadata, covers, popularity, discovering PD full texts | Search/Works/Covers APIs with a descriptive User-Agent [?] | Borrowed books = DRM, **never converted** [D] |
| **User imports** | Modern books the user owns | Upload EPUB (DRM-free) or TXT | User's responsibility; app shows a rights confirmation |

**Genre organisation:** map Library of Congress subjects (`dc:subject`, e.g. "Detective and mystery stories") + Gutenberg bookshelves to our 7 genres (action, horror, thriller, mystery, crime, adventure, fantasy) with a keyword table; unmapped books fall back to a local-LLM classifier. Popularity = Gutenberg download count + Open Library ratings/want-to-read. Publication date = original pub year from Open Library work record (Gutenberg dates are release dates, not publication dates).

**Parsing:** EPUB via `ebooklib` + `lxml` (spine order = chapter order; Standard Ebooks `<section>` = chapter). TXT via heading heuristics (`CHAPTER`, roman numerals) with a manual split editor as backup. Dialogue extracted from quote marks with a tokenizer that handles curly/straight quotes, nested quotes and multi-paragraph speeches (no closing quote until the last paragraph).

---

## 4. Character intelligence pipeline

Runs once per book, offline, results stored in DB and editable by the user.

1. **Structure**: chapters → paragraphs → segments (narration vs quoted dialogue).
2. **BookNLP** (MIT [V], English, `big` model on GPU ≈ 2 min per 100k-token book [D]): character name clustering, coreference, quote → speaker attribution. Note: last commit 2024-07 [V], so it's treated as stable-but-frozen; pinned version.
3. **Local LLM pass** (no paid API; e.g. a Qwen3-class 7–8B instruct model via llama.cpp/Ollama, ~5–6 GB at Q4 [E]): per scene, fix low-confidence attributions (pronouns are BookNLP's weak spot [D]), tag each line's **emotion + intensity**, and write each character's **voice description** (age, gender, accent, timbre, personality).
4. **Scene analysis** (same LLM pass): scene boundaries, location/ambience ("rainy London street, night"), and **SFX cues** anchored to a word ("the door *slammed*").
5. **User review screen**: cast list, merge/split characters, fix speakers, edit voice descriptions. Nothing gets generated until the cast is approved.

**Voice profile per character** (the consistency mechanism):
1. VoiceDesign generates one ~10 s *anchor* clip from the description.
2. For each emotion needed (only the ones the book actually uses), VoiceDesign generates a clip with description + emotion.
3. A speaker-embedding model (e.g. ECAPA via SpeechBrain or WeSpeaker [?, licence to check in Phase 0]) compares each emotion clip to the anchor; below threshold → regenerate.
4. Each approved clip becomes a cached `voice_clone_prompt`. Every line is cloned from the prompt matching its emotion, with a fixed seed. Profile is frozen for the whole book.
5. Narrator gets the same treatment with a calmer bank.

---

## 5. Environmental audio

- **Generation: MOSS-SoundEffect-v2.0** (1.3B DiT flow-matching, 48 kHz, ≤30 s, controllable duration, Apache-2.0) [D]. ~8 GB VRAM [E]. Loops for ambience are made by generating 20–30 s and crossfade-looping.
- **Retrieval: a local asset library first**, generation second, so repeated sounds (footsteps, doors) are consistent and cheap. Library seeded from **Freesound** (free API key; CC0 + CC-BY, attribution stored) and **BBC Sound Effects** (non-commercial, OK for personal use) [?]. Each asset stores licence + source URL.
- **Sync to speech**: segment start/end are known exactly (we generate per segment). Word-level times come from **ctc-forced-aligner** (BSD-2, commit 2026-11-25 [V]) aligning the known text to the generated audio. WhisperX (BSD-2, active [V]) is the fallback.
- **Layers per chapter**: `narration/dialogue` → `SFX` → `ambience` → (optional) `music`.
- **Mixing (FFmpeg)**: ambience ducked under speech with `sidechaincompress`, scene changes with `acrossfade` (default 2 s), SFX placed with `adelay`, final `loudnorm` to an audiobook target (ACX-style: RMS −23 to −18 dB, peak ≤ −3 dB). Ambience default −20 dB below speech, user-adjustable per layer.

---

## 6. System architecture

```
 Phone/Browser
      │
 ┌────▼──────────────┐      REST + SSE (progress)
 │ Next.js (web UI)  │◄──────────────────────────┐
 │ library, cast,    │                           │
 │ timeline editor   │                           │
 └────┬──────────────┘                           │
      │ REST                                     │
 ┌────▼──────────────────────────┐               │
 │ FastAPI (Python)              │───────────────┘
 │ books, projects, segments,    │
 │ voices, jobs, exports         │
 └────┬───────────────┬──────────┘
      │ SQL           │ writes jobs
 ┌────▼─────┐   ┌─────▼──────────────────────────────┐
 │ SQLite   │◄──┤ GPU worker (1 process per GPU)     │
 │ (WAL)    │   │ claims jobs, loads ONE model at a  │
 └──────────┘   │ time: LLM → TTS → SFX → align      │
                │ engines behind adapter interfaces  │
                └─────┬──────────────────────────────┘
                      │
                ┌─────▼─────────────┐
                │ File store        │  /data/projects/<id>/
                │ segments/*.wav    │  chapters/*.wav, exports/*.mp3
                └───────────────────┘
```

**Choices and why:**
- **Next.js** (App Router, TypeScript) for UI only, no business logic. Timeline editor built on **wavesurfer.js** (regions + multitrack) [?, version to pin in Phase 3].
- **Python FastAPI** because all models, BookNLP and alignment are Python.
- **SQLite (WAL) via SQLAlchemy + Alembic.** Single user, one GPU: SQLite is simplest. SQLAlchemy keeps a later move to PostgreSQL a config change.
- **Job queue = a `jobs` table + one worker loop.** No Redis/Celery. One GPU can only run one heavy model at a time anyway. Resume is free: every segment is a row with a status; a restarted worker just picks up `pending`/`failed` rows.
- **Model memory**: the worker loads one model family per job batch and unloads it after (`torch.cuda.empty_cache()`), batching all TTS work for a chapter before switching to SFX.
- **FFmpeg** (system binary, called via `subprocess`) for mixing, fades, loudness, MP3 (libmp3lame, 128 kbps mono / 192 stereo) and WAV export. Later: M4B with chapter markers.

**Engine adapter interface** (same for all TTS engines):
```python
class TTSEngine(Protocol):
    id: str
    capabilities: set[str]        # {"design","clone","instruct","pause_tags"}
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def design_voice(self, description: str, sample_text: str, seed: int) -> Audio: ...
    def build_voice(self, ref: Audio, ref_text: str) -> VoiceHandle: ...
    def synthesize(self, text: str, voice: VoiceHandle, seed: int, **opts) -> Audio: ...
```
SFX engines and aligners get the same pattern (`SFXEngine.generate(prompt, seconds)`, `Aligner.align(audio, text)`). The rest of the app only ever talks to these interfaces, never to a model directly.

### Data model (core tables)

| Table | Key fields |
|---|---|
| `books` | source, source_id, title, author, genres[], pub_year, popularity, rights_note |
| `chapters` | book_id, index, title |
| `segments` | chapter_id, index, kind (narration/dialogue), text, speaker_id, emotion, intensity, status, audio_path, duration, version, word_times(json) |
| `characters` | book_id, name, aliases[], description, is_narrator |
| `voice_profiles` | character_id, engine_id, seed, anchor_path, emotion_bank(json: emotion → clone prompt path), locked |
| `scenes` | chapter_id, start_segment, end_segment, ambience_prompt, ambience_asset_id, gain_db |
| `sfx_cues` | segment_id, word_index, offset_ms, asset_id, gain_db |
| `assets` | kind (sfx/ambience), path, prompt or source_url, licence |
| `projects` | book_id, settings(json: engine, mix levels, export format) |
| `jobs` | type, payload(json), status, progress, error, attempts, created/updated |

---

## 7. Hardware and processing estimates [E]

- **Recommended GPU: 24 GB** (RTX 3090/4090, L4, A10G). 16 GB works for the Qwen3-TTS path only.
- **RAM 32 GB, disk 100 GB+** (model weights ~40 GB for all engines, plus audio).
- **CPU-only**: Kokoro previews, book parsing, mixing, the whole UI/API. Everything else needs the GPU.
- **Book size**: 100k-word novel ≈ 11 h of audio (at ~150 wpm), ≈ 6,000–9,000 segments.
- **Storage per book**: 24 kHz mono 16-bit WAV ≈ 2.9 MB/min → ~2 GB raw segments + mixes; final MP3 ~600 MB.
- **Speed**: unknown until measured. Phase 0 measures real-time factor per engine on the target GPU; that number decides cost per book.

---

## 8. Phased roadmap

Each phase ends with something you can try, and needs your OK before the next starts.

| Phase | Goal | Done when |
|---|---|---|
| **0. Bake-off & setup** | Rent GPU, run Qwen3-TTS, MOSS-TTS v1.5, Kokoro, MOSS-SFX-v2 on one test chapter. Measure VRAM, speed, quality. Check speaker-embedding model licence. | Blind A/B samples + numbers table; primary engine locked |
| **1. Skeleton** | Repo layout, FastAPI + SQLite + Alembic, Next.js shell, jobs table + worker loop, Kokoro adapter | Import a TXT, generate a chapter with Kokoro on CPU, play it in the browser |
| **2. Library** | Standard Ebooks (GitHub) + Gutendex/Gutenberg + Open Library metadata, EPUB/TXT parser, genre mapping, browse/filter UI | Search, import and read 3 books from each source with chapters + dialogue intact |
| **3. Narration MVP** | Primary TTS adapter, narrator voice, chapter generation, resume, MP3/WAV export | Full book narrated in one voice, survives worker restart mid-chapter |
| **4. Characters** | BookNLP + local LLM pass, cast review UI, voice design + emotion banks + similarity check | Multi-voice chapter where every character sounds the same start to end |
| **5. Editor** | Timeline (wavesurfer.js), per-segment regenerate, edit text/speaker/emotion, version history | Fix one wrong line without touching the rest |
| **6. Environmental audio** | Scene/SFX detection, asset library, MOSS-SFX adapter, forced alignment, FFmpeg multi-layer mix with ducking + crossfades | Chapter with ambience + word-synced SFX, adjustable per layer |
| **7. Polish** | Loudness mastering, M4B, second TTS adapter (MOSS), deploy guide, optional Postgres | Release-quality export of a full horror/thriller title |

---

## 9. Decisions (owner, 2026-11-26)

1. **English only** for now.
2. **Rented cloud GPU approved** (pay per hour, switched off when idle).
3. **Personal use only.** Not for sale or publishing. So CC-BY sounds are fine (attribution kept in the asset record), and non-commercial libraries like BBC Sound Effects (RemArc licence: personal/educational/research use [?]) are allowed. If this ever changes to publishing, re-check every asset licence first.

## Sources
- Repos inspected: QwenLM/Qwen3-TTS, OpenMOSS/MOSS-TTS, OpenMOSS/MOSS-TTSD, hexgrad/kokoro, booknlp/booknlp, MahmoudAshraf97/ctc-forced-aligner, m-bain/whisperX, garethbjohnson/gutendex, gutenbergtools/libgutenberg, standardebooks/arthur-conan-doyle_the-hound-of-the-baskervilles
- [Gutenberg robot access policy](https://gutenberg.org/policy/robot_access.html), [Gutenberg offline catalogs](https://www.gutenberg.org/ebooks/offline_catalogs), [Gutenberg mirroring](https://gutenberg.org/help/mirroring.html)
- [Standard Ebooks feeds](https://standardebooks.org/feeds), [Standard Ebooks bulk downloads](https://standardebooks.org/bulk-downloads)
- [Open Library Read API](https://openlibrary.org/dev/docs/api/read), [Open Library borrow FAQ](https://openlibrary.org/help/faq/borrow)
- [MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0), [Qwen3-TTS tech report](https://arxiv.org/abs/2601.15621)
