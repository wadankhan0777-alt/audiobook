#!/usr/bin/env bash
# Phase 0 bake-off. Run on a rented Linux GPU machine (24 GB, CUDA 12.8+ driver):
#   bash bakeoff/run.sh
# Safe to re-run: finished engines are skipped. Delete results/status_<engine>.txt to redo one.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$([ -d /workspace ] && echo /workspace/bakeoff_work || echo "$HOME/bakeoff_work")}"
export BAKEOFF_RESULTS="$WORK/results"
export HF_HOME="$WORK/hf_cache"
export TORCHDYNAMO_DISABLE=1          # MOSS-SFX: skip torch.compile (README workaround)
export UV_LINK_MODE=copy
mkdir -p "$BAKEOFF_RESULTS/logs" "$WORK/src" "$WORK/venvs"

# Pinned to the commits inspected during research (2026-11-26).
MOSS_COMMIT=934d6826b084c46a0d033402174d5f8ac4ed2519
QWEN_COMMIT=022e286b98fbec7e1e916cb940cdf532cd9f488e
KOKORO_COMMIT=dfb907a02bba8152ca444717ca5d78747ccb4bec

say() { echo -e "\n\033[1;36m>>> $*\033[0m"; }

# ---- checks ----
command -v nvidia-smi >/dev/null || { echo "No NVIDIA GPU found. Use a GPU machine."; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$BAKEOFF_RESULTS/gpu.txt"
FREE_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -lt 90 ] && echo "WARNING: only ${FREE_GB} GB free disk here. ~90 GB recommended."

# ---- system packages ----
if command -v apt-get >/dev/null; then
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  say "Installing ffmpeg + espeak-ng"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq ffmpeg espeak-ng git >/dev/null
fi
command -v uv >/dev/null || { say "Installing uv"; python3 -m pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$WORK/src/MOSS-TTS" ]; then
  say "Fetching MOSS-TTS source"
  git clone -q https://github.com/OpenMOSS/MOSS-TTS.git "$WORK/src/MOSS-TTS"
fi
git -C "$WORK/src/MOSS-TTS" checkout -q "$MOSS_COMMIT"

# ---- per-engine install + run ----
install_env() {  # $1 = engine
  local V="$WORK/venvs/$1"
  [ -x "$V/bin/python" ] || uv venv -q --python 3.12 "$V"
  local PIP="uv pip install -q --python $V/bin/python"
  $PIP pip soundfile numpy
  case "$1" in
    kokoro)   $PIP --torch-backend cu128 "kokoro @ git+https://github.com/hexgrad/kokoro@$KOKORO_COMMIT" ;;
    qwen3)    $PIP --torch-backend cu128 "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS@$QWEN_COMMIT" ;;
    moss)     (cd "$WORK/src/MOSS-TTS" && $PIP --torch-backend cu128 -e ".[torch-runtime]") ;;
    moss_sfx) (cd "$WORK/src/MOSS-TTS/moss_soundeffect_v2" && $PIP --extra-index-url https://download.pytorch.org/whl/cu128 \
                 --index-strategy unsafe-best-match -e ".[torch-cu128]") ;;
  esac
}

run_engine() {  # $1 = engine, $2 = script
  local E="$1" LOG="$BAKEOFF_RESULTS/logs/$1.log"
  if [ "$(cat "$BAKEOFF_RESULTS/status_$E.txt" 2>/dev/null)" = "ok" ]; then say "$E already done, skipping"; return; fi
  say "[$E] installing (log: $LOG)"
  if ! install_env "$E" >"$LOG" 2>&1; then echo "install_failed" >"$BAKEOFF_RESULTS/status_$E.txt"; say "[$E] INSTALL FAILED, see log"; return; fi
  sed -i "/\"engine\": \"$E\"/d" "$BAKEOFF_RESULTS/metrics.jsonl" 2>/dev/null
  say "[$E] running (first run downloads the model weights, can take a while)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >"$BAKEOFF_RESULTS/vram_raw_$E.txt" &
  local SAMPLER=$!
  (cd "$HERE" && "$WORK/venvs/$E/bin/python" "$2") >>"$LOG" 2>&1
  local RC=$?
  kill "$SAMPLER" 2>/dev/null
  awk 'max<$1{max=$1} END{printf "%.1f", max/1024}' "$BAKEOFF_RESULTS/vram_raw_$E.txt" >"$BAKEOFF_RESULTS/vram_$E.txt"
  if [ $RC = 0 ]; then echo ok >"$BAKEOFF_RESULTS/status_$E.txt"; say "[$E] done"
  else echo failed >"$BAKEOFF_RESULTS/status_$E.txt"; say "[$E] FAILED, last lines:"; tail -15 "$LOG"; fi
}

run_engine kokoro   bench_kokoro.py
run_engine qwen3    bench_qwen3.py
run_engine moss     bench_moss.py
run_engine moss_sfx bench_sfx.py

python3 "$HERE/collect.py" "$BAKEOFF_RESULTS"
(cd "$WORK" && python3 -m zipfile -c bakeoff_results.zip results/listen results/logs results/metrics.jsonl \
   results/gpu.txt results/key_DONT_OPEN_UNTIL_YOU_PICK.json 2>/dev/null)
echo "Zip of everything: $WORK/bakeoff_results.zip"
