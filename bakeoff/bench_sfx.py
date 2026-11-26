"""MOSS-SoundEffect-v2.0: ambience + one-shot effects."""
import soundfile as sf
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline

from common import RESULTS, SFX, Timer, record, seed_all, torch_peak_gb

E = "moss_sfx"
OUT = RESULTS / "raw" / E
OUT.mkdir(parents=True, exist_ok=True)

with Timer() as t:
    pipe = MossSoundEffectPipeline.from_pretrained("OpenMOSS-Team/MOSS-SoundEffect-v2.0",
                                                   torch_dtype=torch.bfloat16, device="cuda")
record(E, step="load", seconds=round(t.s, 1))

for name, prompt, seconds in SFX:
    seed_all()
    with Timer() as t:
        audio = pipe(prompt=prompt, seconds=seconds, num_inference_steps=100, cfg_scale=4.0)
    path = OUT / f"{name}.wav"
    pipe.save_audio(audio, str(path))
    dur = sf.info(str(path)).duration
    record(E, step="sfx", item=name, gen_s=round(t.s, 1), audio_s=round(dur, 1),
           rtf=round(t.s / dur, 3), torch_peak_gb=torch_peak_gb())
