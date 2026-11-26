"""MOSS: design each voice (+ emotion refs) with MOSS-VoiceGenerator, then clone every line with MOSS-TTS-v1.5."""
import importlib.util

import torch
from transformers import AutoModel, AutoProcessor

from common import (LONG_TEXT, REF_TEXT, RESULTS, SEGMENTS, VOICES, EMOTION_STYLE, Timer, concat,
                    emotions_needed, free_gpu, record, save, seed_all, torch_peak_gb)

# From the MOSS README: disable the broken cuDNN SDPA backend, keep the others.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

E = "moss"
OUT = RESULTS / "raw" / E
DEV, DTYPE = "cuda", torch.bfloat16
ATTN = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


def load(repo, **proc_kw):
    proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True, **proc_kw)
    proc.audio_tokenizer = proc.audio_tokenizer.to(DEV)
    model = AutoModel.from_pretrained(repo, trust_remote_code=True, attn_implementation=ATTN,
                                      dtype=DTYPE).to(DEV).eval()
    return proc, model


def generate(proc, model, msg, **gen_kw):
    batch = proc([[msg]], mode="generation")
    with torch.no_grad():
        out = model.generate(input_ids=batch["input_ids"].to(DEV),
                             attention_mask=batch["attention_mask"].to(DEV), **gen_kw)
    audio = next(iter(proc.decode(out))).audio_codes_list[0]
    return audio, proc.model_config.sampling_rate


# 1) Voice design. Sampling values = model card defaults for MOSS-VoiceGenerator.
VG_KW = dict(audio_temperature=1.5, audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1)
with Timer() as t:
    proc, model = load("OpenMOSS-Team/MOSS-VoiceGenerator", normalize_inputs=True)
record(E, step="design_load", seconds=round(t.s, 1))

refs = {}
for spk, desc in VOICES.items():
    for emo in emotions_needed(spk):
        seed_all()
        with Timer() as t:
            audio, sr = generate(proc, model, proc.build_user_message(
                text=REF_TEXT[emo], instruction=desc + EMOTION_STYLE[emo]), **VG_KW)
        path = OUT / "refs" / f"{spk}_{emo}.wav"
        dur = save(path, audio, sr)
        refs[(spk, emo)] = str(path.resolve())
        record(E, step="design", item=f"{spk}_{emo}", gen_s=round(t.s, 2), audio_s=round(dur, 2))
record(E, step="design_done", torch_peak_gb=torch_peak_gb())
del proc, model
free_gpu()
torch.cuda.reset_peak_memory_stats()

# 2) Clone with MOSS-TTS-v1.5 (defaults per model card: temp 1.7, top_p 0.8, top_k 25).
with Timer() as t:
    proc, model = load("OpenMOSS-Team/MOSS-TTS-v1.5")
record(E, step="clone_load", seconds=round(t.s, 1))

parts, gen_total, audio_total = [], 0.0, 0.0
for i, (spk, emo, text) in enumerate(SEGMENTS):
    seed_all()
    with Timer() as t:
        audio, sr = generate(proc, model, proc.build_user_message(
            text=text, language="English", reference=[refs[(spk, emo)]]), max_new_tokens=2048)
    dur = save(OUT / "segments" / f"{i:02d}_{spk}_{emo}.wav", audio, sr)
    parts.append(audio)
    gen_total += t.s
    audio_total += dur
save(OUT / "scene.wav", concat(parts, sr), sr)
record(E, step="scene", gen_s=round(gen_total, 1), audio_s=round(audio_total, 1),
       rtf=round(gen_total / audio_total, 3), torch_peak_gb=torch_peak_gb())

seed_all()
with Timer() as t:
    audio, sr = generate(proc, model, proc.build_user_message(
        text=LONG_TEXT, language="English", reference=[refs[("narrator", "neutral")]]),
        max_new_tokens=4096)
dur = save(OUT / "long.wav", audio, sr)
record(E, step="long", gen_s=round(t.s, 1), audio_s=round(dur, 1), rtf=round(t.s / dur, 3),
       torch_peak_gb=torch_peak_gb())
