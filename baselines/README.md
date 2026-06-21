# Baseline Model Setup

This directory isolates the external baseline repositories and normalizes their outputs for the shared evaluator.

For a file-by-file sharing checklist and plain-language walkthrough, see:

```text
voicegrad/EVALUATION_PIPELINE_SHARE_GUIDE.md
```

## Directory Layout

```text
voicegrad/baselines/
  repos/
    StarGAN-VC/
    autovc/
    ppg-vc/
    Diff-VC/
  manifests/
    conversion_manifest.csv
    target_references.csv
  shared/
    source_by_pair/
    stargan_arctic_4spk/
  stargan-vc/generated_wavs/
  autovc/generated_wavs/
  ppg-vc/generated_wavs/
  diff-vc/generated_wavs/
```

Every baseline should eventually produce:

```text
voicegrad/baselines/<model>/generated_wavs/<src>_to_<tgt>/<utt_id>.wav
```

That is the only output contract required by `voicegrad/evaluation_pipeline/evaluate_models.py`.

## Command Shells

Most examples below include both shell forms:

- macOS/Linux, WSL, or Git Bash: use the `bash` blocks.
- Windows PowerShell: use the `powershell` blocks.

On Windows, prefer `python` instead of `python3`. The StarGAN-VC recipes are shell scripts (`.sh` files), so they need WSL or Git Bash even when launched from Windows Terminal.

## Rebuild The Workspace

Run this whenever the VoiceGrad test protocol changes:

```bash
python3 voicegrad/baselines/scripts/prepare_workspace.py
```

```powershell
python voicegrad/baselines/scripts/prepare_workspace.py
```

## Runtime Environment

The baseline repos depend on PyTorch and older audio packages. Use a clean environment instead of the project default Python if possible:

```bash
conda env create -f voicegrad/baselines/environment.yml
conda activate voicegrad-baselines
```

```powershell
conda env create -f voicegrad/baselines/environment.yml
conda activate voicegrad-baselines
```

Check the current setup:

```bash
python3 voicegrad/baselines/scripts/check_setup.py
```

```powershell
python voicegrad/baselines/scripts/check_setup.py
```

```powershell
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/check_setup.py
```

If scripted Google Drive downloads fail, use the browser URLs in:

```text
voicegrad/baselines/manifests/checkpoint_urls.csv
```

Download each asset manually and place it at the listed `target_path`.

It creates:

- `manifests/conversion_manifest.csv`: the 384 closed-set conversions.
- `manifests/target_references.csv`: one target reference utterance per target speaker for one-shot baselines.
- `shared/source_by_pair/<src>_to_<tgt>/*.wav`: source utterances grouped by conversion pair.
- `shared/stargan_arctic_4spk/training` and `test`: StarGAN-compatible speaker folders.
- Empty `generated_wavs` folders for StarGAN-VC, AutoVC, PPG-VC, and Diff-VC.

## StarGAN-VC

Repo:

```text
voicegrad/baselines/repos/StarGAN-VC
```

Train with the Arctic-compatible recipe:

```bash
cd voicegrad/baselines/repos/StarGAN-VC

& "C:\Program Files\Git\bin\bash.exe" ./recipes/run_train_arctic_4spk.sh `
  -g 0 `
  -d ../../shared/stargan_arctic_4spk/training `
  -e voicegrad_arctic_4spk
```

From Windows Terminal PowerShell, launch the same recipe through WSL or Git Bash:

```powershell
cd voicegrad/baselines/repos/StarGAN-VC

bash ./recipes/run_train_arctic_4spk.sh `
  -g 0 `
  -d ../../shared/stargan_arctic_4spk/training `
  -e voicegrad_arctic_4spk
```

Then convert the test set:

```bash
./recipes/run_test_arctic_4spk.sh \
  -g 0 \
  -d ../../shared/stargan_arctic_4spk/test \
  -e voicegrad_arctic_4spk
```

```powershell
bash ./recipes/run_test_arctic_4spk.sh `
  -g 0 `
  -d ../../shared/stargan_arctic_4spk/test `
  -e voicegrad_arctic_4spk
```

If your shell prompt shows both `voicegrad-baselines` and `.venv`, deactivate the virtualenv first so the recipe's plain `python` command uses the conda environment:

```bash
deactivate
conda activate voicegrad-baselines
```

```powershell
deactivate
conda activate voicegrad-baselines
```

If a previous StarGAN run failed before writing `.h5` features, remove the stale generated dump files before rerunning training:

```bash
rm -rf dump/arctic_4spk/feat dump/arctic_4spk/norm_feat dump/arctic_4spk/stat.pkl
```

```powershell
Remove-Item -Recurse -Force dump/arctic_4spk/feat, dump/arctic_4spk/norm_feat, dump/arctic_4spk/stat.pkl
```

StarGAN writes pair folders like `bdl2clb`. Normalize them to the evaluator layout:

```bash
python3 ../../scripts/normalize_stargan_outputs.py \
  --stargan-out-root out/arctic_4spk/voicegrad_arctic_4spk/0/hifigan.v1
```

```powershell
python ../../scripts/normalize_stargan_outputs.py `
  --stargan-out-root out/arctic_4spk/voicegrad_arctic_4spk/0/hifigan.v1
```

Adjust the `--stargan-out-root` checkpoint/vocoder path if the repo writes a different checkpoint number.

Important: StarGAN-VC also needs a compatible vocoder under its `pwg/egs/...` path, as described in its README.

## AutoVC

Repo:

```text
voicegrad/baselines/repos/autovc
```

AutoVC is notebook/checkpoint oriented, so this project uses a wrapper:

```text
voicegrad/baselines/scripts/run_autovc_batch.py
```

Use the AutoVC README to install/download:

- AutoVC checkpoint.
- Speaker encoder checkpoint, usually `3000000-BL.ckpt`.
- WaveNet or HiFi-GAN vocoder checkpoint.

Place the default checkpoints here:

```text
voicegrad/baselines/repos/autovc/autovc.ckpt
voicegrad/baselines/repos/autovc/3000000-BL.ckpt
voicegrad/baselines/repos/autovc/checkpoint_step001000000_ema.pth
```

Install the original WaveNet vocoder package if you want the paper-style AutoVC synthesis path:

```bash
python3 -m pip install wavenet_vocoder
```

```powershell
python -m pip install wavenet_vocoder
```

Run one smoke-test pair:

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

Run all conversions:

```bash
python3 voicegrad/baselines/scripts/run_autovc_batch.py
```

```powershell
python voicegrad/baselines/scripts/run_autovc_batch.py
```

The default `--vocoder wavenet` path needs the WaveNet checkpoint above. For a fast plumbing check without the WaveNet package/checkpoint, you can synthesize lower-quality audio with:

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

The Griffin-Lim fallback is useful for verifying paths and metric scripts, but do not use it for final model comparisons unless you explicitly report that AutoVC used Griffin-Lim instead of its pretrained neural vocoder.

Outputs are written to:

```text
voicegrad/baselines/autovc/generated_wavs/<src>_to_<tgt>/<utt_id>.wav
```

## PPG-VC

Repo:

```text
voicegrad/baselines/repos/ppg-vc
```

Download the pretrained PPG-VC assets from the repo README. The wrapper expects:

- PPG model files in `repos/ppg-vc/conformer_ppg_model/en_conformer_ctc_att/`.
- Speaker encoder checkpoint at `repos/ppg-vc/speaker_encoder/ckpt/pretrained_bak_5805000.pt`.
- HiFi-GAN files in the repo’s expected `vocoders` location.
- A PPG2Mel config and checkpoint path passed on the command line.

You can download the upstream Google Drive folder with:

```bash
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/download_checkpoints.py ppg-vc
```

```powershell
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/download_checkpoints.py ppg-vc
```

Then copy the selected checkpoint into a stable path, for example:

```text
voicegrad/baselines/repos/ppg-vc/checkpts/bneSeq2seqMoL-vctk-libritts460-oneshot/best_loss_step_304000.pth
```

Run all pairs:

```bash
python3 voicegrad/baselines/scripts/run_ppg_vc_batch.py \
  --config voicegrad/baselines/repos/ppg-vc/checkpts/bneSeq2seqMoL-vctk-libritts460-oneshot/seq2seq_mol_ppg2mel_vctk_libri_oneshotvc_r4_normMel_v2.yaml \
  --checkpoint voicegrad/baselines/repos/ppg-vc/checkpts/bneSeq2seqMoL-vctk-libritts460-oneshot/best_loss_step_304000.pth
```

```powershell
python voicegrad/baselines/scripts/run_ppg_vc_batch.py `
  --config voicegrad/baselines/repos/ppg-vc/checkpts/bneSeq2seqMoL-vctk-libritts460-oneshot/seq2seq_mol_ppg2mel_vctk_libri_oneshotvc_r4_normMel_v2.yaml `
  --checkpoint voicegrad/baselines/repos/ppg-vc/checkpts/bneSeq2seqMoL-vctk-libritts460-oneshot/best_loss_step_304000.pth
```

For a smoke test on one pair:

```bash
python3 voicegrad/baselines/scripts/run_ppg_vc_batch.py \
  --pairs bdl_to_clb \
  --config <config.yaml> \
  --checkpoint <checkpoint.pth>
```

```powershell
python voicegrad/baselines/scripts/run_ppg_vc_batch.py `
  --pairs bdl_to_clb `
  --config <config.yaml> `
  --checkpoint <checkpoint.pth>
```

Note: the upstream `convert_from_wav.py` hardcodes `device = 'cuda'`, so PPG-VC currently expects a CUDA environment unless you patch that repo file.

## Diff-VC

Repo:

```text
voicegrad/baselines/repos/Diff-VC
```

Download the pretrained files from the Diff-VC README and place them at:

```text
voicegrad/baselines/repos/Diff-VC/checkpts/vc/vc_libritts_wodyn.pt
voicegrad/baselines/repos/Diff-VC/checkpts/vocoder/config.json
voicegrad/baselines/repos/Diff-VC/checkpts/vocoder/generator
voicegrad/baselines/repos/Diff-VC/checkpts/spk_encoder/pretrained.pt
```

The original Diff-VC speaker-encoder Google Drive link may be unavailable.
Use the packaged Resemblyzer checkpoint instead:

```bash
conda run -n voicegrad-baselines python -m pip install resemblyzer
conda run -n voicegrad-baselines python -c "import pathlib, shutil, resemblyzer; src=pathlib.Path(resemblyzer.__file__).resolve().parent/'pretrained.pt'; dst=pathlib.Path('voicegrad/baselines/repos/Diff-VC/checkpts/spk_encoder/pretrained.pt'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); print(src, '->', dst)"
```

```powershell
conda run -n voicegrad-baselines python -m pip install resemblyzer
conda run -n voicegrad-baselines python -c "import pathlib, shutil, resemblyzer; src=pathlib.Path(resemblyzer.__file__).resolve().parent/'pretrained.pt'; dst=pathlib.Path('voicegrad/baselines/repos/Diff-VC/checkpts/spk_encoder/pretrained.pt'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); print(src, '->', dst)"
```

Or download the LibriTTS model set with:

```bash
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/download_checkpoints.py diff-vc
```

```powershell
conda run -n voicegrad-baselines python voicegrad/baselines/scripts/download_checkpoints.py diff-vc
```

The downloader copies the Resemblyzer speaker encoder and uses Google Drive only for the Diff-VC vocoder and VC model.

Run:

```bash
python3 voicegrad/baselines/scripts/run_diff_vc_batch.py
```

```powershell
python voicegrad/baselines/scripts/run_diff_vc_batch.py
```

For one pair:

```bash
python3 voicegrad/baselines/scripts/run_diff_vc_batch.py --pairs bdl_to_clb
```

```powershell
python voicegrad/baselines/scripts/run_diff_vc_batch.py --pairs bdl_to_clb
```

## Evaluate Everything

After generating outputs for any subset of baselines:

```bash
bash voicegrad/baselines/evaluate_all.sh
```

```powershell
bash voicegrad/baselines/evaluate_all.sh
```

For the full metric pass including ASR CER and DNSMOS pMOS, run from the baseline environment:

```bash
conda activate voicegrad-baselines
bash voicegrad/baselines/evaluate_all_full.sh
```

```powershell
conda activate voicegrad-baselines
bash voicegrad/baselines/evaluate_all_full.sh
```

The comparison CSVs are written to:

```text
voicegrad/evaluation_pipeline/comparison_eval/
```
