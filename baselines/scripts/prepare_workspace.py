#!/usr/bin/env python3
"""Prepare shared inputs and output folders for baseline VC models.

This script does not train or run a model. It creates the filesystem contract
used by the baseline wrappers and by the evaluation pipeline:

    voicegrad/baselines/<model>/generated_wavs/<src>_to_<tgt>/<utt_id>.wav
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import pandas as pd


BASELINE_NAMES = ("stargan-vc", "autovc", "ppg-vc", "diff-vc")


def rel_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    target = os.path.relpath(src.resolve(), dst.parent.resolve())
    try:
        dst.symlink_to(target)
        return
    except FileExistsError:
        return
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise

    try:
        os.link(src, dst)
    except FileExistsError:
        return
    except OSError:
        shutil.copy2(src, dst)


def load_manifest_from_voicegrad(voicegrad_root: Path) -> pd.DataFrame:
    meta_path = voicegrad_root / "checkpoints" / "ckpt1000_closedset_test_result" / "generated_meta.csv"
    if meta_path.exists():
        df = pd.read_csv(meta_path, encoding="utf-8-sig")
        return df[["src_spk", "tgt_spk", "utt_id"]].drop_duplicates()

    generated_root = voicegrad_root / "checkpoints" / "ckpt1000_closedset_test_result" / "generated_wavs"
    rows = []
    for wav_path in sorted(generated_root.glob("*/*.wav")):
        pair = wav_path.parent.name
        if "_to_" not in pair:
            continue
        src_spk, tgt_spk = pair.split("_to_", 1)
        rows.append({"src_spk": src_spk, "tgt_spk": tgt_spk, "utt_id": wav_path.stem})
    if not rows:
        raise SystemExit(f"No VoiceGrad test outputs found under {generated_root}")
    return pd.DataFrame(rows).drop_duplicates()


def choose_reference_wavs(voicegrad_root: Path, speakers: list[str], eval_utt_ids: set[str]) -> list[dict[str, str]]:
    rows = []
    wav_root = voicegrad_root / "data" / "wav"
    for speaker in speakers:
        candidates = sorted((wav_root / speaker / "wav").glob("*.wav"))
        if not candidates:
            raise SystemExit(f"No wav files found for speaker {speaker}: {wav_root / speaker / 'wav'}")
        non_eval = [p for p in candidates if p.stem not in eval_utt_ids]
        chosen = non_eval[0] if non_eval else candidates[0]
        rows.append({"speaker": speaker, "reference_wav": str(chosen.resolve()), "utt_id": chosen.stem})
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_pair_source_dirs(baselines_root: Path, voicegrad_root: Path, manifest: pd.DataFrame) -> None:
    source_root = baselines_root / "shared" / "source_by_pair"
    wav_root = voicegrad_root / "data" / "wav"
    for row in manifest.itertuples(index=False):
        src_wav = wav_root / row.src_spk / "wav" / f"{row.utt_id}.wav"
        if not src_wav.exists():
            raise SystemExit(f"Missing source wav: {src_wav}")
        dst = source_root / f"{row.src_spk}_to_{row.tgt_spk}" / f"{row.utt_id}.wav"
        rel_symlink(src_wav, dst)


def prepare_stargan_dataset(baselines_root: Path, voicegrad_root: Path, manifest: pd.DataFrame, speakers: list[str]) -> None:
    dataset_root = baselines_root / "shared" / "stargan_arctic_4spk"
    eval_utt_ids = set(manifest["utt_id"].astype(str))
    wav_root = voicegrad_root / "data" / "wav"

    for speaker in speakers:
        speaker_wavs = sorted((wav_root / speaker / "wav").glob("*.wav"))
        for wav_path in speaker_wavs:
            split = "test" if wav_path.stem in eval_utt_ids else "training"
            rel_symlink(wav_path, dataset_root / split / speaker / wav_path.name)


def prepare_output_dirs(baselines_root: Path, manifest: pd.DataFrame) -> None:
    for model_name in BASELINE_NAMES:
        for row in manifest.itertuples(index=False):
            out_dir = baselines_root / model_name / "generated_wavs" / f"{row.src_spk}_to_{row.tgt_spk}"
            out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voicegrad-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baselines-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    voicegrad_root = args.voicegrad_root.resolve()
    baselines_root = args.baselines_root.resolve()
    manifest = load_manifest_from_voicegrad(voicegrad_root).sort_values(["src_spk", "tgt_spk", "utt_id"])
    speakers = sorted(set(manifest["src_spk"]) | set(manifest["tgt_spk"]))
    eval_utt_ids = set(manifest["utt_id"].astype(str))

    manifest_rows = []
    for row in manifest.itertuples(index=False):
        src_wav = voicegrad_root / "data" / "wav" / row.src_spk / "wav" / f"{row.utt_id}.wav"
        tgt_wav = voicegrad_root / "data" / "wav" / row.tgt_spk / "wav" / f"{row.utt_id}.wav"
        manifest_rows.append(
            {
                "src_spk": row.src_spk,
                "tgt_spk": row.tgt_spk,
                "utt_id": row.utt_id,
                "src_wav": str(src_wav.resolve()),
                "tgt_wav": str(tgt_wav.resolve()),
                "pair": f"{row.src_spk}_to_{row.tgt_spk}",
                "expected_output": f"generated_wavs/{row.src_spk}_to_{row.tgt_spk}/{row.utt_id}.wav",
            }
        )

    write_csv(
        baselines_root / "manifests" / "conversion_manifest.csv",
        manifest_rows,
        ["src_spk", "tgt_spk", "utt_id", "src_wav", "tgt_wav", "pair", "expected_output"],
    )
    write_csv(
        baselines_root / "manifests" / "target_references.csv",
        choose_reference_wavs(voicegrad_root, speakers, eval_utt_ids),
        ["speaker", "reference_wav", "utt_id"],
    )

    prepare_pair_source_dirs(baselines_root, voicegrad_root, manifest)
    prepare_stargan_dataset(baselines_root, voicegrad_root, manifest, speakers)
    prepare_output_dirs(baselines_root, manifest)

    print(f"Prepared {len(manifest_rows)} conversions across {len(speakers)} speakers.")
    print(f"Wrote {baselines_root / 'manifests' / 'conversion_manifest.csv'}")
    print(f"Wrote {baselines_root / 'manifests' / 'target_references.csv'}")
    print(f"Prepared pair source dirs under {baselines_root / 'shared' / 'source_by_pair'}")
    print(f"Prepared StarGAN dataset under {baselines_root / 'shared' / 'stargan_arctic_4spk'}")


if __name__ == "__main__":
    main()
