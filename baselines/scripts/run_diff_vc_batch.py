#!/usr/bin/env python3
"""Run Diff-VC conversion for every row in the shared manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch


def load_inferencer(repo_dir: Path, vc_checkpoint: Path, hifigan_dir: Path, spk_encoder_checkpoint: Path, output_path: Path):
    sys.path.insert(0, str(repo_dir.resolve()))
    sys.path.insert(0, str((repo_dir / "hifi-gan").resolve()))
    sys.path.insert(0, str((repo_dir / "speaker_encoder").resolve()))

    import params  # type: ignore
    from env import AttrDict  # type: ignore
    from inference import Inferencer  # type: ignore
    from model import DiffVC  # type: ignore
    from models import Generator as HiFiGAN  # type: ignore
    from encoder import inference as spk_encoder  # type: ignore

    use_gpu = torch.cuda.is_available()
    generator = DiffVC(
        params.n_mels,
        params.channels,
        params.filters,
        params.heads,
        params.layers,
        params.kernel,
        params.dropout,
        params.window_size,
        params.enc_dim,
        params.spk_dim,
        params.use_ref_t,
        params.dec_dim,
        params.beta_min,
        params.beta_max,
    )
    if use_gpu:
        generator = generator.cuda()
        generator.load_state_dict(torch.load(vc_checkpoint))
    else:
        generator.load_state_dict(torch.load(vc_checkpoint, map_location="cpu"))
    generator.eval()

    with (hifigan_dir / "config.json").open() as f:
        h = AttrDict(json.load(f))
    hifigan = HiFiGAN(h).cuda() if use_gpu else HiFiGAN(h)
    hifigan_state = torch.load(hifigan_dir / "generator", map_location=None if use_gpu else "cpu")
    hifigan.load_state_dict(hifigan_state["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()

    spk_encoder.load_model(spk_encoder_checkpoint, device="cuda" if use_gpu else "cpu")
    output_path.mkdir(parents=True, exist_ok=True)
    return Inferencer(generator, spk_encoder, hifigan, str(output_path), use_gpu=use_gpu)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    baselines_root = Path(__file__).resolve().parents[1]
    repo_dir = baselines_root / "repos" / "Diff-VC"
    parser.add_argument("--repo-dir", type=Path, default=repo_dir)
    parser.add_argument("--manifest", type=Path, default=baselines_root / "manifests" / "conversion_manifest.csv")
    parser.add_argument("--target-refs", type=Path, default=baselines_root / "manifests" / "target_references.csv")
    parser.add_argument("--vc-checkpoint", type=Path, default=repo_dir / "checkpts" / "vc" / "vc_libritts_wodyn.pt")
    parser.add_argument("--hifigan-dir", type=Path, default=repo_dir / "checkpts" / "vocoder")
    parser.add_argument("--spk-encoder-checkpoint", type=Path, default=repo_dir / "checkpts" / "spk_encoder" / "pretrained.pt")
    parser.add_argument("--work-output", type=Path, default=baselines_root / "runs" / "diff-vc")
    parser.add_argument("--dest-root", type=Path, default=baselines_root / "diff-vc" / "generated_wavs")
    parser.add_argument("--pairs", nargs="*", help="Optional pair filter, e.g. bdl_to_clb slt_to_rms")
    parser.add_argument("--n-timesteps", type=int, default=30)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    refs = pd.read_csv(args.target_refs).set_index("speaker")["reference_wav"].to_dict()
    if args.pairs:
        manifest = manifest[manifest["pair"].isin(args.pairs)]

    inferencer = load_inferencer(
        args.repo_dir,
        args.vc_checkpoint,
        args.hifigan_dir,
        args.spk_encoder_checkpoint,
        args.work_output,
    )

    copied = 0
    for row in manifest.itertuples(index=False):
        dest = args.dest_root / row.pair / f"{row.utt_id}.wav"
        if args.skip_existing and dest.exists():
            print(f"skip existing: {row.pair}/{row.utt_id}.wav")
            continue
        target_ref = refs[row.tgt_spk]
        generated_path = Path(
            inferencer.infer(
                row.src_wav,
                target_ref,
                n_timesteps=args.n_timesteps,
                return_output_path=True,
                sr=22050,
            )
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_path, dest)
        copied += 1
        print(f"{copied}: {row.pair}/{row.utt_id}.wav")

    print(f"Copied {copied} Diff-VC outputs to {args.dest_root}")


if __name__ == "__main__":
    main()
