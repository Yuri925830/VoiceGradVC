#!/usr/bin/env bash
# Run the lightweight comparison pass for every model folder.

# Stop immediately if any command fails, if an unset variable is used, or if a pipeline fails.
set -euo pipefail

# Move from voicegrad/baselines/ to the project root so all relative paths below resolve correctly.
cd "$(dirname "$0")/../.."

# Run the shared evaluator.
# --compute-audio computes paper-style MCD/LFC for generated WAVs.
# Existing CER and pMOS CSVs are loaded if present, but ASR/DNSMOS are not recomputed here.
# Each --model argument maps a display name to a result directory under voicegrad/.
python3 voicegrad/evaluation_pipeline/evaluate_models.py \
  --compute-audio \
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result \
  --model StarGAN-VC=baselines/stargan-vc \
  --model AutoVC=baselines/autovc \
  --model PPG-VC=baselines/ppg-vc \
  --model Diff-VC=baselines/diff-vc
