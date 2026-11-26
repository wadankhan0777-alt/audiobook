"""Qwen3-TTS: design each voice (+ emotion refs) with VoiceDesign, then clone every line with Base."""
import torch
from qwen_tts import Qwen3TTSModel

from common import (LONG_TEXT, REF_TEXT, RESULTS, SEGMENTS, VOICES, EMOTION_STYLE, Timer, concat,
                    emotions_needed, free_gpu, record, save, seed_all, torch_peak_gb)

E = "qwen3"
OUT = RESULTS / "raw" / E
LOAD = dict(device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

# 1) Voice design: one reference clip per (character, emotion)
with Timer() as t:
    design = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", **LOAD)
record(E, step="design_load", seconds=round(t.s, 1))

refs = {}
for spk, desc in VOICES.items():
    for emo in emotions_needed(spk):
        seed_all()
        with Timer() as t:
            wavs, sr = design.generate_voice_design(
                text=REF_TEXT[emo], language="English", instruct=desc + EMOTION_STYLE[emo])
        dur = save(OUT / "refs" / f"{spk}_{emo}.wav", wavs[0], sr)
        refs[(spk, emo)] = (wavs[0], sr)
        record(E, step="design", item=f"{spk}_{emo}", gen_s=round(t.s, 2), audio_s=round(dur, 2))
record(E, step="design_done", torch_peak_gb=torch_peak_gb())
del design
free_gpu()
torch.cuda.reset_peak_memory_stats()

# 2) Clone: reusable prompt per (character, emotion), then synthesize each segment
with Timer() as t:
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", **LOAD)
record(E, step="clone_load", seconds=round(t.s, 1))

prompts = {k: base.create_voice_clone_prompt(ref_audio=v, ref_text=REF_TEXT[k[1]]) for k, v in refs.items()}

parts, gen_total, audio_total = [], 0.0, 0.0
for i, (spk, emo, text) in enumerate(SEGMENTS):
    seed_all()
    with Timer() as t:
        wavs, sr = base.generate_voice_clone(text=text, language="English",
                                             voice_clone_prompt=prompts[(spk, emo)])
    dur = save(OUT / "segments" / f"{i:02d}_{spk}_{emo}.wav", wavs[0], sr)
    parts.append(wavs[0])
    gen_total += t.s
    audio_total += dur
save(OUT / "scene.wav", concat(parts, sr), sr)
record(E, step="scene", gen_s=round(gen_total, 1), audio_s=round(audio_total, 1),
       rtf=round(gen_total / audio_total, 3), torch_peak_gb=torch_peak_gb())

seed_all()
with Timer() as t:
    wavs, sr = base.generate_voice_clone(text=LONG_TEXT, language="English",
                                         voice_clone_prompt=prompts[("narrator", "neutral")])
dur = save(OUT / "long.wav", wavs[0], sr)
record(E, step="long", gen_s=round(t.s, 1), audio_s=round(dur, 1), rtf=round(t.s / dur, 3),
       torch_peak_gb=torch_peak_gb())
