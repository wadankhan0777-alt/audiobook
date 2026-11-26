"""Shared test content + helpers for the Phase 0 bake-off.

Every bench script runs in its own venv, so this file only needs numpy + soundfile.
Test text: The Hound of the Baskervilles, ch. 2 (US public domain, via Standard Ebooks).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

RESULTS = Path(os.environ.get("BAKEOFF_RESULTS", "results"))
SEED = 1234
GAP_SECONDS = 0.35  # silence between segments in the scene file

# Voice descriptions used by the voice-design models (Qwen VoiceDesign, MOSS VoiceGenerator).
VOICES = {
    "narrator": "Mature male narrator, warm and clear British English, measured pace, "
                "rich low-mid voice, calm storytelling tone.",
    "holmes": "Male, about 35, crisp upper-class British English, sharp, precise and "
              "confident, slightly dry and detached.",
    "mortimer": "Male, about 30, earnest country doctor, British English, gentle tenor "
                "voice with nervous energy.",
}

# Added to the description when designing an emotion reference clip.
EMOTION_STYLE = {
    "neutral": "",
    "fearful": " Speaking with growing fear and alarm, breath unsteady.",
    "whisper": " Hushed, trembling near-whisper, full of dread.",
}

# What each designed reference clip says (also passed as the clone transcript).
REF_TEXT = {
    "neutral": "It was a cold evening in late autumn, and the fog lay thick upon the "
               "streets of London as I made my way home.",
    "fearful": "There was something out there in the dark, I swear it, something "
               "watching us from beyond the gate.",
    "whisper": "Listen. Do you hear it? It is coming closer, across the moor.",
}

# Hand-segmented scene (Phase 0 has no NLP; this is what the pipeline will produce later).
SEGMENTS = [
    ("holmes", "neutral", "I must thank you,"),
    ("narrator", "neutral", "said Sherlock Holmes,"),
    ("holmes", "neutral", "for calling my attention to a case which certainly presents some "
                          "features of interest. This article, you say, contains all the public facts?"),
    ("mortimer", "neutral", "It does."),
    ("holmes", "neutral", "Then let me have the private ones."),
    ("narrator", "neutral", "He leaned back, put his fingertips together, and assumed his most "
                            "impassive and judicial expression."),
    ("mortimer", "fearful", "I whisked round and had just time to catch a glimpse of something which "
                            "I took to be a large black calf passing at the head of the drive. So excited "
                            "and alarmed was he that I was compelled to go down to the spot where the "
                            "animal had been and look around for it."),
    ("mortimer", "neutral", "But one false statement was made by Barrymore at the inquest. He said that "
                            "there were no traces upon the ground round the body. He did not observe any. "
                            "But I did, some little distance off, but fresh and clear."),
    ("holmes", "neutral", "Footprints?"),
    ("mortimer", "neutral", "Footprints."),
    ("holmes", "neutral", "A man's or a woman's?"),
    ("narrator", "neutral", "Dr. Mortimer looked strangely at us for an instant, and his voice sank "
                            "almost to a whisper as he answered."),
    ("mortimer", "whisper", "Mr. Holmes, they were the footprints of a gigantic hound!"),
]

# One long narrator passage generated in a single call (long-form stability test, ~300 words).
LONG_TEXT = (
    "It was at my advice that Sir Charles was about to go to London. His heart was, I knew, affected, "
    "and the constant anxiety in which he lived, however chimerical the cause of it might be, was "
    "evidently having a serious effect upon his health. I thought that a few months among the "
    "distractions of town would send him back a new man. Mr. Stapleton, a mutual friend who was much "
    "concerned at his state of health, was of the same opinion. At the last instant came this terrible "
    "catastrophe. On the night of Sir Charles's death Barrymore the butler, who made the discovery, sent "
    "Perkins the groom on horseback to me, and as I was sitting up late I was able to reach Baskerville "
    "Hall within an hour of the event. I checked and corroborated all the facts which were mentioned at "
    "the inquest. I followed the footsteps down the yew alley, I saw the spot at the moor-gate where he "
    "seemed to have waited, I remarked the change in the shape of the prints after that point, I noted "
    "that there were no other footsteps save those of Barrymore on the soft gravel, and finally I "
    "carefully examined the body, which had not been touched until my arrival. Sir Charles lay on his "
    "face, his arms out, his fingers dug into the ground, and his features convulsed with some strong "
    "emotion to such an extent that I could hardly have sworn to his identity. There was certainly no "
    "physical injury of any kind. But one false statement was made by Barrymore at the inquest. He said "
    "that there were no traces upon the ground round the body. He did not observe any. But I did, some "
    "little distance off, but fresh and clear."
)

# Sound-effect prompts: (name, prompt, seconds)
SFX = [
    ("ambience_moor_night", "Wind sweeping across an empty moor at night, distant owl, soft rustling "
                            "grass, eerie and desolate.", 20),
    ("door_slam", "A heavy old wooden door slams shut in a stone hallway, with a short echo.", 3),
    ("footsteps_gravel", "Slow footsteps walking on soft gravel, one person, close up.", 6),
    ("hound_howl", "A huge dog howling in the distance across open moorland at night.", 8),
]


def emotions_needed(speaker):
    return sorted({e for s, e, _ in SEGMENTS if s == speaker} | {"neutral"})


def to_mono_np(audio):
    """torch tensor or numpy -> float32 numpy, shape (T,) or (T, C)."""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    a = np.asarray(audio, dtype=np.float32).squeeze()
    if a.ndim == 2 and a.shape[0] <= 2:  # (C, T) -> (T, C)
        a = a.T
    return a


def save(path, audio, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = to_mono_np(audio)
    sf.write(str(path), a, sr)
    return len(a) / sr


def concat(parts, sr, gap=GAP_SECONDS):
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    out = []
    for p in parts:
        a = to_mono_np(p)
        if a.ndim == 2:
            a = a.mean(axis=1)
        out += [a, silence]
    return np.concatenate(out[:-1]) if out else silence


class Timer:
    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        self.s = time.perf_counter() - self.t


def torch_peak_gb():
    try:
        import torch
        return round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def record(engine, **fields):
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = {"engine": engine, **fields}
    with open(RESULTS / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def seed_all(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
