#!/usr/bin/env python3
"""Copy StarGAN-VC outputs into the common evaluator layout.

StarGAN-VC writes pair folders as `<src>2<tgt>`. The evaluator expects
`<src>_to_<tgt>`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "manifests" / "conversion_manifest.csv")
    parser.add_argument(
        "--stargan-out-root",
        type=Path,
        required=True,
        help="Directory containing StarGAN pair folders, e.g. repos/StarGAN-VC/out/arctic_4spk/<exp>/<ckpt>/hifigan.v1",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "stargan-vc" / "generated_wavs",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    copied = 0
    missing = []
    for row in manifest.itertuples(index=False):
        src = args.stargan_out_root / f"{row.src_spk}2{row.tgt_spk}" / f"{row.utt_id}.wav"
        dst = args.dest_root / f"{row.src_spk}_to_{row.tgt_spk}" / f"{row.utt_id}.wav"
        if not src.exists():
            missing.append(str(src))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} StarGAN outputs to {args.dest_root}")
    if missing:
        print(f"Missing {len(missing)} expected StarGAN outputs. First missing file:")
        print(missing[0])


if __name__ == "__main__":
    main()
