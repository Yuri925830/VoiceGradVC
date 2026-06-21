#!/usr/bin/env python3
"""Run ppg-vc conversion for every pair in the shared manifest.

This wrapper expects the ppg-vc repo checkpoints to already be installed in the
repo-specific paths required by `convert_from_wav.py`.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def copy_pair_outputs(tmp_output_dir: Path, dest_pair_dir: Path, utt_ids: list[str]) -> tuple[int, list[str]]:
    copied = 0
    missing = []
    dest_pair_dir.mkdir(parents=True, exist_ok=True)
    for utt_id in utt_ids:
        matches = sorted(glob.glob(str(tmp_output_dir / f"vc_{utt_id}_ref_*_step*.wav")))
        if not matches:
            missing.append(utt_id)
            continue
        shutil.copy2(matches[-1], dest_pair_dir / f"{utt_id}.wav")
        copied += 1
    return copied, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    baselines_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-dir", type=Path, default=baselines_root / "repos" / "ppg-vc")
    parser.add_argument("--manifest", type=Path, default=baselines_root / "manifests" / "conversion_manifest.csv")
    parser.add_argument("--target-refs", type=Path, default=baselines_root / "manifests" / "target_references.csv")
    parser.add_argument("--config", type=Path, required=True, help="PPG-VC yaml config path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PPG-VC ppg2mel checkpoint path")
    parser.add_argument("--dest-root", type=Path, default=baselines_root / "ppg-vc" / "generated_wavs")
    parser.add_argument("--pairs", nargs="*", help="Optional pair filter, e.g. bdl_to_clb slt_to_rms")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    refs = pd.read_csv(args.target_refs).set_index("speaker")["reference_wav"].to_dict()
    pairs = sorted(manifest["pair"].unique())
    if args.pairs:
        pairs = [p for p in pairs if p in set(args.pairs)]

    for pair in pairs:
        pair_rows = manifest[manifest["pair"] == pair]
        tgt_spk = pair_rows.iloc[0]["tgt_spk"]
        ref_wav = refs[tgt_spk]
        src_wav_dir = baselines_root / "shared" / "source_by_pair" / pair
        tmp_output_dir = baselines_root / "runs" / "ppg-vc" / pair
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "convert_from_wav.py",
            "--ppg2mel_model_train_config",
            str(args.config.resolve()),
            "--ppg2mel_model_file",
            str(args.checkpoint.resolve()),
            "--src_wav_dir",
            str(src_wav_dir.resolve()),
            "--ref_wav_path",
            str(Path(ref_wav).resolve()),
            "--output_dir",
            str(tmp_output_dir.resolve()),
            "--device",
            args.device,
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=args.repo_dir, check=True)
            copied, missing = copy_pair_outputs(
                tmp_output_dir,
                args.dest_root / pair,
                pair_rows["utt_id"].astype(str).tolist(),
            )
            print(f"{pair}: copied {copied}, missing {len(missing)}")
            if missing:
                print(f"{pair}: first missing utterance {missing[0]}")


if __name__ == "__main__":
    main()
