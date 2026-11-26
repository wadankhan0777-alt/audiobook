"""Blind-label the samples and print one summary table. Stdlib only."""
import json
import random
import shutil
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
RAW, LISTEN = RESULTS / "raw", RESULTS / "listen"
TTS = ["kokoro", "qwen3", "moss"]
BOOK_HOURS = 11  # ~100k-word novel

if LISTEN.exists():
    shutil.rmtree(LISTEN)
LISTEN.mkdir(parents=True)

key, rng = {}, random.SystemRandom()
for test in ["scene", "long"]:
    engines = [e for e in TTS if (RAW / e / f"{test}.wav").exists()]
    rng.shuffle(engines)
    for letter, e in zip("ABC", engines):
        shutil.copy(RAW / e / f"{test}.wav", LISTEN / f"{test}_{letter}.wav")
        key[f"{test}_{letter}"] = e
for f in sorted((RAW / "moss_sfx").glob("*.wav")):
    shutil.copy(f, LISTEN / f"sfx_{f.name}")
(RESULTS / "key_DONT_OPEN_UNTIL_YOU_PICK.json").write_text(json.dumps(key, indent=2))

rows = [json.loads(l) for l in (RESULTS / "metrics.jsonl").read_text().splitlines()] \
    if (RESULTS / "metrics.jsonl").exists() else []
vram = {}
for f in RESULTS.glob("vram_*.txt"):
    vram[f.stem[5:]] = f.read_text().strip()
status = {}
for f in RESULTS.glob("status_*.txt"):
    status[f.stem[7:]] = f.read_text().strip()

print("\n================ BAKE-OFF SUMMARY (paste this back to Claude) ================")
for e in TTS + ["moss_sfx"]:
    r = {x["step"]: x for x in rows if x["engine"] == e}
    line = f"{e:9s} status={status.get(e, 'not run'):8s} gpu_peak={vram.get(e, '?')}GB"
    for step in ["scene", "long"]:
        if step in r:
            line += f" | {step}: rtf={r[step]['rtf']} audio={r[step]['audio_s']}s"
    if "long" in r:
        line += f" | ~{round(r['long']['rtf'] * BOOK_HOURS, 1)} GPU-h per 11h book"
    if e == "moss_sfx":
        s = [x for x in rows if x["engine"] == e and x["step"] == "sfx"]
        if s:
            line += f" | {len(s)} clips, avg rtf={round(sum(x['rtf'] for x in s) / len(s), 2)}"
    print(line)
gpu = (RESULTS / "gpu.txt").read_text().strip() if (RESULTS / "gpu.txt").exists() else "?"
print(f"GPU: {gpu}")
print("==============================================================================")
print(f"Listen to the files in {LISTEN} . Pick your favourite scene_ and long_ letter.")
