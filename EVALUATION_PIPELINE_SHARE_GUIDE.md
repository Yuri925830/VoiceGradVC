# Evaluation Pipeline Share Guide

This document explains is shared with teammates, what each file does, and what still needs to be supplied before all baseline comparisons can run.

## Short Answer

Shared:

```text
voicegrad/evaluation_pipeline/
voicegrad/baselines/README.md
voicegrad/baselines/environment.yml
voicegrad/baselines/evaluate_all.sh
voicegrad/baselines/evaluate_all_full.sh
voicegrad/baselines/manifests/
voicegrad/baselines/scripts/
```

Also clone the baseline repository links or use the already cloned `voicegrad/baselines/repos/` folder

Generated/cache folders:

```text
voicegrad/evaluation_pipeline/__pycache__/
voicegrad/baselines/scripts/__pycache__/
voicegrad/evaluation_pipeline/comparison_eval/
```

`comparison_eval/` is generated output. Contain current result tables.

## What This Pipeline Does

The pipeline has two jobs:

1. Normalize baseline model outputs into one shared folder convention:

```text
voicegrad/baselines/<model>/generated_wavs/<src>_to_<tgt>/<utt_id>.wav
```

2. Evaluate/compare those WAVs using the same metrics as the VoiceGrad notebooks:

- MCD: lower is better.
- LFC: higher is better.
- CER: lower is better.
- pMOS/DNSMOS: higher is better.

## Main Evaluation Folder

### `voicegrad/evaluation_pipeline/evaluate_models.py`

This is the main CLI evaluator.

It can:

- Load existing metric CSVs from each model folder.
- Compute paper-style MCD/LFC from generated WAVs.
- Optionally compute CER using wav2vec2.
- Optionally compute pMOS using `speechmos.dnsmos`.
- Merge all models into one comparison table.
- Rank models by metric direction.

Important functions inside this script:

```text
parse_model_specs()
```

Parses command-line model arguments such as `StarGAN-VC=baselines/stargan-vc`.

```text
load_existing_details()
```

Loads existing `mcd_lfc_detail.csv`, `cer_detail.csv`, and `pmos_dnsmos_detail.csv` from a model folder.

```text
iter_generated_wavs()
```

Scans `generated_wavs/<src>_to_<tgt>/*.wav` and turns files into evaluation rows.

```text
compute_mcd_lfc()
```

Computes paper-style MCD and LFC using WORLD, SPTK mel-cepstra, and DTW alignment.

```text
compute_audio_details()
```

Runs `compute_mcd_lfc()` for every generated WAV in one model folder.

```text
load_ref_map()
```

Parses `cmuarctic.data.txt` into an utterance-ID-to-transcript dictionary.

```text
compute_cer_details()
```

Runs wav2vec2 ASR, normalizes the hypothesis text, and computes CER against the transcript.

```text
compute_pmos_details()
```

Runs DNSMOS through `speechmos` and stores the predicted MOS value.

```text
merge_existing_and_computed()
```

Combines existing CSV metrics with newly computed metrics. Existing values stay authoritative unless `--prefer-computed` is passed.

```text
summarize()
```

Computes overall and speaker-pair means, confidence intervals, and counts.

```text
main()
```

Parses CLI flags, runs requested metric computations, writes detail CSVs, writes summary CSVs, and prints output paths.

Important command:

```bash
python3 voicegrad/evaluation_pipeline/evaluate_models.py \
  --compute-audio \
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result \
  --model StarGAN-VC=baselines/stargan-vc \
  --model AutoVC=baselines/autovc \
  --model PPG-VC=baselines/ppg-vc \
  --model Diff-VC=baselines/diff-vc
```

```powershell
python voicegrad/evaluation_pipeline/evaluate_models.py `
  --compute-audio `
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result `
  --model StarGAN-VC=baselines/stargan-vc `
  --model AutoVC=baselines/autovc `
  --model PPG-VC=baselines/ppg-vc `
  --model Diff-VC=baselines/diff-vc
```

For full metrics:

```bash
conda activate voicegrad-baselines

python3 voicegrad/evaluation_pipeline/evaluate_models.py \
  --compute-audio \
  --compute-cer \
  --compute-pmos \
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result \
  --model StarGAN-VC=baselines/stargan-vc \
  --model AutoVC=baselines/autovc \
  --model PPG-VC=baselines/ppg-vc \
  --model Diff-VC=baselines/diff-vc
```

```powershell
conda activate voicegrad-baselines

python voicegrad/evaluation_pipeline/evaluate_models.py `
  --compute-audio `
  --compute-cer `
  --compute-pmos `
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result `
  --model StarGAN-VC=baselines/stargan-vc `
  --model AutoVC=baselines/autovc `
  --model PPG-VC=baselines/ppg-vc `
  --model Diff-VC=baselines/diff-vc
```

### `voicegrad/evaluation_pipeline/EVALUATION.md`

This is the usage document for the evaluator.

Share this with teammates who only need to run the comparison once outputs already exist.

### `voicegrad/evaluation_pipeline/eval_requirements.txt`

This is the pip-style dependency list for evaluation metrics.

The conda environment in `voicegrad/baselines/environment.yml` is usually easier for baseline work.

### `voicegrad/evaluation_pipeline/comparison_eval/`

This folder contains generated comparison CSVs.

Files inside it are outputs, not source code:

```text
model_metric_detail.csv
model_metric_summary.csv
model_metric_summary_by_pair.csv
metric_manifest.csv
```

Share these only if you want teammates to see your current result tables.

## Baseline Folder

### `voicegrad/baselines/README.md`

This is the main baseline setup guide.

It explains:

- baseline folder layout
- checkpoint locations
- model-specific commands
- evaluation commands

### `voicegrad/baselines/environment.yml`

This defines the conda environment used for baseline inference and full evaluation.

Create it with:

```bash
conda env create -f voicegrad/baselines/environment.yml
conda activate voicegrad-baselines
```

```powershell
conda env create -f voicegrad/baselines/environment.yml
conda activate voicegrad-baselines
```

### `voicegrad/baselines/evaluate_all.sh`

This evaluates all models using MCD/LFC and any existing CER/pMOS CSVs.

It does not force ASR/DNSMOS recomputation.

### `voicegrad/baselines/evaluate_all_full.sh`

This evaluates all models with:

- MCD/LFC
- CER
- pMOS

Run this only after activating `voicegrad-baselines`, because it needs ASR and DNSMOS dependencies.

## Baseline Manifests

### `voicegrad/baselines/manifests/conversion_manifest.csv`

This is the core conversion protocol.

Each row says:

- source speaker
- target speaker
- utterance ID
- source WAV path
- target/reference WAV path
- expected baseline output path

This file defines the `384` closed-set conversions.

### `voicegrad/baselines/manifests/target_references.csv`

This maps each target speaker to one reference WAV.

It is used by one-shot/few-shot models such as PPG-VC and Diff-VC.

### `voicegrad/baselines/manifests/checkpoint_urls.csv`

This lists checkpoint download URLs and target paths.

Google Drive scripted downloads may fail, so this CSV is the manual fallback.

## Baseline Scripts

### `voicegrad/baselines/scripts/prepare_workspace.py`

This rebuilds the baseline workspace from the VoiceGrad closed-set test protocol.

It creates:

- `conversion_manifest.csv`
- `target_references.csv`
- source-by-pair folders
- StarGAN train/test folders
- empty output folders for each baseline

Run it after changing the test split:

```bash
python3 voicegrad/baselines/scripts/prepare_workspace.py
```

```powershell
python voicegrad/baselines/scripts/prepare_workspace.py
```

### `voicegrad/baselines/scripts/check_setup.py`

This checks whether the baseline environment, checkpoint files, and generated outputs are ready.

Run it often:

```bash
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/check_setup.py
```

```powershell
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/check_setup.py
```

### `voicegrad/baselines/scripts/download_checkpoints.py`

This attempts scripted checkpoint downloads for checkpoint groups with known Google Drive IDs.

If it fails because Google Drive blocks scripted downloads, use `checkpoint_urls.csv` manually.

### `voicegrad/baselines/scripts/run_ppg_vc_batch.py`

This runs PPG-VC for every conversion pair in the manifest.

It calls the upstream `ppg-vc/convert_from_wav.py` script and copies outputs into the common evaluator layout.

### `voicegrad/baselines/scripts/run_diff_vc_batch.py`

This runs Diff-VC for every row in the manifest.

It loads the Diff-VC model, speaker encoder, and HiFi-GAN vocoder, then writes outputs into the common evaluator layout.

### `voicegrad/baselines/scripts/run_autovc_batch.py`

This runs AutoVC for every row in the manifest.

It replaces AutoVC's two notebooks with one command-line path:

- load source WAVs from `conversion_manifest.csv`
- compute AutoVC mel spectrograms
- compute source and target speaker embeddings
- run the pretrained AutoVC generator
- synthesize waveform audio with WaveNet, or with Griffin-Lim for quick checks
- write outputs into the common evaluator layout

Smoke test one conversion:

```bash
python3 voicegrad/baselines/scripts/run_autovc_batch.py \
  --pairs bdl_to_clb \
  --limit 1
```

```powershell
python voicegrad/baselines/scripts/run_autovc_batch.py `
  --pairs bdl_to_clb `
  --limit 1
```

Run all AutoVC conversions:

```bash
python3 voicegrad/baselines/scripts/run_autovc_batch.py
```

```powershell
python voicegrad/baselines/scripts/run_autovc_batch.py
```

For path/debug checks without the WaveNet vocoder package or checkpoint:

```bash
python3 voicegrad/baselines/scripts/run_autovc_batch.py \
  --pairs bdl_to_clb \
  --limit 1 \
  --vocoder griffin-lim
```

```powershell
python voicegrad/baselines/scripts/run_autovc_batch.py `
  --pairs bdl_to_clb `
  --limit 1 `
  --vocoder griffin-lim
```

Use the default WaveNet vocoder for final comparisons when possible. Griffin-Lim outputs are lower quality and should be clearly reported if used.

### `voicegrad/baselines/scripts/normalize_stargan_outputs.py`

StarGAN writes pair folders like:

```text
bdl2clb/
```

The evaluator expects:

```text
bdl_to_clb/
```

This script copies StarGAN outputs into the evaluator layout.

## External Repositories

These are external codebases, not project-owned pipeline code:

```text
voicegrad/baselines/repos/StarGAN-VC/
voicegrad/baselines/repos/autovc/
voicegrad/baselines/repos/ppg-vc/
voicegrad/baselines/repos/Diff-VC/
```

You can share them as cloned folders, but it may be cleaner to share the repo URLs and let teammates clone them:

- StarGAN-VC: `https://github.com/kamepong/StarGAN-VC`
- AutoVC: `https://github.com/auspicious3000/autovc`
- PPG-VC: `https://github.com/liusongxiang/ppg-vc`
- Diff-VC: `https://github.com/trinhtuanvubk/Diff-VC`

One local change was made to:

```text
voicegrad/baselines/repos/ppg-vc/convert_from_wav.py
```

The change adds a `--device auto|cuda|cpu` option so the script is not hardcoded to CUDA.

If teammates clone PPG-VC fresh, they need this patch or they need to run on CUDA.

## What Still Needs To Be Implemented Or Supplied

### Still Missing Checkpoints

The VoiceGrad checkpoints in `voicegrad/checkpoints/` are not enough for baseline models.

Each baseline requires its own architecture-specific checkpoints:

- PPG-VC needs its PPG2Mel checkpoint.
- Diff-VC needs its VC checkpoint, speaker encoder, and vocoder generator.
- AutoVC needs AutoVC, speaker encoder, and vocoder checkpoints.
- StarGAN-VC needs a trained StarGAN model and compatible vocoder assets.

Use:

```bash
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/check_setup.py
```

to see exactly which files are missing.

### Still Missing Baseline Outputs

Currently the baseline output folders exist, but they do not contain generated WAVs yet:

```text
voicegrad/baselines/stargan-vc/generated_wavs/
voicegrad/baselines/autovc/generated_wavs/
voicegrad/baselines/ppg-vc/generated_wavs/
voicegrad/baselines/diff-vc/generated_wavs/
```

Until those folders contain WAVs, the evaluator can only report the VoiceGrad row.

### AutoVC Adapter

The AutoVC batch wrapper is implemented at:

```text
voicegrad/baselines/scripts/run_autovc_batch.py
```

It still needs the pretrained AutoVC checkpoint, speaker encoder checkpoint, and vocoder checkpoint before it can produce final comparison WAVs.

## Recommended Sharing Package

If sending a zip or branch to teammates, include:

```text
voicegrad/EVALUATION_PIPELINE_SHARE_GUIDE.md
voicegrad/evaluation_pipeline/
voicegrad/evaluation_pipeline/EVALUATION.md
voicegrad/baselines/README.md
voicegrad/baselines/environment.yml
voicegrad/baselines/evaluate_all.sh
voicegrad/baselines/evaluate_all_full.sh
voicegrad/baselines/manifests/
voicegrad/baselines/scripts/
```

Optionally include:

```text
voicegrad/evaluation_pipeline/comparison_eval/
```

Do not include:

```text
__pycache__/
*.pyc
```
