# VoiceGrad Private Git Transfer

This repo is configured to track source code, notebooks, data manifests, evaluation scripts, and the small closed-set VoiceGrad result folder.

Large generated/model artifacts are intentionally ignored:

- `checkpoints/*.pt`
- baseline model checkpoints
- generated baseline WAVs
- StarGAN `dump/`, `logs/`, `model/`, and `out/`
- nested external repos under `baselines/repos/`
- `hifi-gan/`

## Create The Private Repo

Create an empty private repo in GitHub/GitLab first, then run this from the local machine:

```bash
cd /Users/fionapuhringer/Documents/Auslandssemester/courses/COSE362/team_project/voicegrad
git init
git add .
git commit -m "Add VoiceGrad evaluation pipeline"
git branch -M main
git remote add origin <YOUR_PRIVATE_REPO_URL>
git push -u origin main
```

## Pull On The Compute Machine

```bash
git clone <YOUR_PRIVATE_REPO_URL> voicegrad
cd voicegrad
conda env create -f baselines/environment.yml
conda activate voicegrad-baselines
bash baselines/scripts/clone_baseline_repos.sh
python baselines/scripts/prepare_workspace.py
python baselines/scripts/check_setup.py
```

The baseline repos are cloned separately because they are external Git repositories. Local compatibility changes are stored in `baselines/patches/` and applied by `clone_baseline_repos.sh`.

## Move Large Artifacts Separately

Use one of these for large files that Git should not carry:

```bash
rsync -avh --progress checkpoints/ user@remote:/path/to/voicegrad/checkpoints/
rsync -avh --progress data/ user@remote:/path/to/voicegrad/data/
```

If the compute machine can access the same cloud storage, downloading the baseline checkpoints there is usually cleaner than pushing them through Git.

## Monitor StarGAN Training

From the `voicegrad` directory:

```bash
python baselines/scripts/monitor_stargan_training.py --once
python baselines/scripts/monitor_stargan_training.py --interval 30
```

The monitor prints feature-file counts, normalized feature counts, latest epoch/minibatch from the log, and checkpoint count.
