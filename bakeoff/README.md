# Phase 0 bake-off

Runs Kokoro, Qwen3-TTS (VoiceDesign → Base clone), MOSS (VoiceGenerator → MOSS-TTS-v1.5 clone) and MOSS-SoundEffect-v2 on the same Hound of the Baskervilles scene, then blind-labels the results.

What each TTS engine makes:
- `scene_X.wav`: 13 lines, 3 characters (narrator, Holmes, Dr. Mortimer). Mortimer has neutral, fearful and whisper versions of his voice (the emotion-bank approach from ARCHITECTURE.md §4). Listen for: does Mortimer still sound like the same man when scared?
- `long_X.wav`: ~300 words of narration in one call (long-form stability).

Plus `sfx_*.wav`: moor ambience, door slam, gravel footsteps, distant hound.

Run on a 24 GB GPU machine with ~90 GB disk: `bash bakeoff/run.sh`. Re-running skips finished engines. Output goes to `/workspace/bakeoff_work/results/` (or `~/bakeoff_work/results/`).

Pick your favourite letters **before** opening `key_DONT_OPEN_UNTIL_YOU_PICK.json`.
