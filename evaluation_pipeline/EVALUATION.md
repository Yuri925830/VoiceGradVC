# VoiceGrad Baseline Evaluation

This folder now has a standalone comparison script:

For the full sharing checklist and peer-facing walkthrough, see:

```text
voicegrad/EVALUATION_PIPELINE_SHARE_GUIDE.md
```

```bash
python3 voicegrad/evaluation_pipeline/evaluate_models.py
```

```powershell
python voicegrad/evaluation_pipeline/evaluate_models.py
```

By default it aggregates the existing VoiceGrad result folder:

```text
voicegrad/checkpoints/ckpt1000_closedset_test_result
```

The script writes:

```text
voicegrad/evaluation_pipeline/comparison_eval/model_metric_detail.csv
voicegrad/evaluation_pipeline/comparison_eval/model_metric_summary.csv
voicegrad/evaluation_pipeline/comparison_eval/model_metric_summary_by_pair.csv
voicegrad/evaluation_pipeline/comparison_eval/metric_manifest.csv
```

## Baseline Folder Layout

Put each baseline model output in the same structure as VoiceGrad:

```text
voicegrad/baselines/
  stargan-vc/generated_wavs/bdl_to_clb/arctic_b0508.wav
  autovc/generated_wavs/bdl_to_clb/arctic_b0508.wav
  ppg-vc/generated_wavs/bdl_to_clb/arctic_b0508.wav
  diff-vc/generated_wavs/bdl_to_clb/arctic_b0508.wav
```

The expected speaker-pair folder format is:

```text
<src_spk>_to_<tgt_spk>/<utt_id>.wav
```

Target reference WAVs are read from:

```text
voicegrad/data/wav/<tgt_spk>/wav/<utt_id>.wav
```

## Compare All Models

If each baseline folder already contains metric CSVs named like the VoiceGrad output:

```text
mcd_lfc_detail.csv
cer_detail.csv
pmos_dnsmos_detail.csv
```

run:

```bash
python3 voicegrad/evaluation_pipeline/evaluate_models.py \
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result \
  --model StarGAN-VC=baselines/stargan-vc \
  --model AutoVC=baselines/autovc \
  --model PPG-VC=baselines/ppg-vc \
  --model Diff-VC=baselines/diff-vc
```

```powershell
python voicegrad/evaluation_pipeline/evaluate_models.py `
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result `
  --model StarGAN-VC=baselines/stargan-vc `
  --model AutoVC=baselines/autovc `
  --model PPG-VC=baselines/ppg-vc `
  --model Diff-VC=baselines/diff-vc
```

To compute paper-style MCD and LFC from `generated_wavs` instead:

```bash
python3 -m pip install -r voicegrad/evaluation_pipeline/eval_requirements.txt

python3 voicegrad/evaluation_pipeline/evaluate_models.py \
  --compute-audio \
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result \
  --model StarGAN-VC=baselines/stargan-vc \
  --model AutoVC=baselines/autovc \
  --model PPG-VC=baselines/ppg-vc \
  --model Diff-VC=baselines/diff-vc
```

```powershell
python -m pip install -r voicegrad/evaluation_pipeline/eval_requirements.txt

python voicegrad/evaluation_pipeline/evaluate_models.py `
  --compute-audio `
  --model VoiceGrad-DPM-BNF=checkpoints/ckpt1000_closedset_test_result `
  --model StarGAN-VC=baselines/stargan-vc `
  --model AutoVC=baselines/autovc `
  --model PPG-VC=baselines/ppg-vc `
  --model Diff-VC=baselines/diff-vc
```

This uses the same method as the VoiceGrad notebooks: WORLD F0/spectral envelope extraction, SPTK mel-cepstra, DTW alignment, MCD, and aligned log-F0 correlation.

By default, existing `mcd_lfc_detail.csv` values remain authoritative and computed MCD/LFC fill only missing values. Add `--prefer-computed` if you want the newly computed audio metrics to replace existing CSV values for models that already have them.

To compute CER and pMOS too:

```bash
conda activate voicegrad-baselines

python3 voicegrad/evaluation_pipeline/evaluate_models.py \
  --compute-audio \
  --compute-cer \
  --compute-pmos \
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
  --model StarGAN-VC=baselines/stargan-vc `
  --model AutoVC=baselines/autovc `
  --model PPG-VC=baselines/ppg-vc `
  --model Diff-VC=baselines/diff-vc
```

CER uses `facebook/wav2vec2-large-960h-lv60-self` by default and pMOS uses `speechmos.dnsmos`, matching the local VoiceGrad evaluation notebooks. Computed metric CSVs are written into each model folder by default; add `--no-write-model-metrics` for a read-only dry run.

## Metrics

| Metric | Column | Better | Notes |
| --- | --- | --- | --- |
| MCD | `mcd_db` | lower | Spectral distortion between converted and target speech. |
| LFC | `lfc` | higher | Log-F0 contour correlation for intonation similarity. |
| CER | `cer` | lower | Linguistic content preservation using ASR transcript output. |
| pMOS / MOS | `pmos` or `mos` | higher | Predicted or human naturalness/audio-quality score. |

CER and pMOS/MOS are aggregated when the model result folder has `cer_detail.csv` and `pmos_dnsmos_detail.csv`, or computed when `--compute-cer` / `--compute-pmos` are passed.
