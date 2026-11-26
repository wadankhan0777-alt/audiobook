"""Kokoro-82M: fixed British voicepacks, no emotion control (baseline / preview engine)."""
from kokoro import KPipeline

from common import LONG_TEXT, RESULTS, SEGMENTS, Timer, concat, record, save, seed_all, torch_peak_gb

E = "kokoro"
OUT = RESULTS / "raw" / E
SR = 24000
VOICE = {"narrator": "bm_george", "holmes": "bm_lewis", "mortimer": "bm_daniel"}

with Timer() as t:
    pipe = KPipeline(lang_code="b")  # b = British English
record(E, step="load", seconds=round(t.s, 1))


def speak(text, voice):
    return concat([audio for _, _, audio in pipe(text, voice=voice)], SR, gap=0.0)


parts, gen_total, audio_total = [], 0.0, 0.0
for i, (spk, emo, text) in enumerate(SEGMENTS):
    seed_all()
    with Timer() as t:
        a = speak(text, VOICE[spk])
    audio_total += save(OUT / "segments" / f"{i:02d}_{spk}_{emo}.wav", a, SR)
    gen_total += t.s
    parts.append(a)
save(OUT / "scene.wav", concat(parts, SR), SR)
record(E, step="scene", gen_s=round(gen_total, 1), audio_s=round(audio_total, 1),
       rtf=round(gen_total / audio_total, 3), torch_peak_gb=torch_peak_gb())

with Timer() as t:
    a = speak(LONG_TEXT, VOICE["narrator"])
dur = save(OUT / "long.wav", a, SR)
record(E, step="long", gen_s=round(t.s, 1), audio_s=round(dur, 1), rtf=round(t.s / dur, 3),
       torch_peak_gb=torch_peak_gb())
