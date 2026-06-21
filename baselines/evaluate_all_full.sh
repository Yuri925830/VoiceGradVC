#!/usr/bin/env bash
# Run the full comparison pass for every model folder.
# This includes the expensive metrics: ASR CER and DNSMOS pMOS.

# Stop immediately if any command fails, if an unset variable is used, or if a pipeline fails.
set -euo pipefail

# Move from voicegrad/baselines/ to the project root so all relative paths below resolve correctly.
cd "$(dirname "$0")/../.."

# Run the shared evaluator.
# --compute-audio computes paper-style MCD/LFC.
# --compute-cer runs wav2vec2 ASR and computes character error rate.
# --compute-pmos runs speechmos DNSMOS and extracts overall predicted MOS.
# Each --model argument maps a display name to a result directory under voicegrad/.
python3 voicegrad/evaluation_pipeline/evaluate_models.py \
  --compute-audio \
  --compute-cer \
  --compute-pmos \
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result \
  --model StarGAN-VC=baselines/stargan-vc \
  --model AutoVC=baselines/autovc \
  --model PPG-VC=baselines/ppg-vc \
  --model Diff-VC=baselines/diff-vc
