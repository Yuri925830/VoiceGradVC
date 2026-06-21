#!/usr/bin/env python3
"""Preflight checks for baseline model execution."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd


BASELINE_ASSETS = {
    "ppg-vc": {
        "required": [
            "repos/ppg-vc/conformer_ppg_model/en_conformer_ctc_att/config.yaml",
            "repos/ppg-vc/conformer_ppg_model/en_conformer_ctc_att/24epoch.pth",
            "repos/ppg-vc/speaker_encoder/ckpt/pretrained_bak_5805000.pt",
            "repos/ppg-vc/vocoders/vctk_24k10ms/config.json",
            "repos/ppg-vc/vocoders/vctk_24k10ms/g_02830000",
        ],
        "missing_by_default": [
            "repos/ppg-vc/checkpts/bneSeq2seqMoL-vctk-libritts460-oneshot/best_loss_step_304000.pth",
        ],
    },
    "diff-vc": {
        "required": [
            "repos/Diff-VC/checkpts/vocoder/config.json",
        ],
        "missing_by_default": [
            "repos/Diff-VC/checkpts/vc/vc_libritts_wodyn.pt",
            "repos/Diff-VC/checkpts/vocoder/generator",
            "repos/Diff-VC/checkpts/spk_encoder/pretrained.pt",
        ],
    },
    "autovc": {
        "required": [],
        "missing_by_default": [
            "repos/autovc/3000000-BL.ckpt",
            "repos/autovc/autovc.ckpt",
            "repos/autovc/checkpoint_step001000000_ema.pth",
        ],
    },
    "stargan-vc": {
        "required": [],
        "missing_by_default": [
            "repos/StarGAN-VC/model/arctic_4spk/voicegrad_arctic_4spk/model_config.json",
            "repos/StarGAN-VC/pwg/egs/arctic_4spk_flen64ms_fshift8ms/voc1",
        ],
    },
}

PYTHON_IMPORTS = {
    "shared": ["numpy", "pandas", "soundfile", "scipy", "tqdm", "yaml"],
    "torch_models": ["torch"],
    "ppg-vc": ["pyworld", "resampy"],
    "stargan-vc": ["h5py", "joblib", "sklearn"],
    "autovc": ["librosa"],
    "autovc_wavenet": ["wavenet_vocoder"],
    "diff-vc": ["librosa"],
}


def check_import(module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def print_status(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"[{mark}] {label}{suffix}")


def count_outputs(baselines_root: Path, model_name: str, manifest: pd.DataFrame) -> int:
    count = 0
    for row in manifest.itertuples(index=False):
        if (baselines_root / model_name / row.expected_output).exists():
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.baselines_root.resolve()
    manifest_path = root / "manifests" / "conversion_manifest.csv"

    print("Python imports")
    for group, modules in PYTHON_IMPORTS.items():
        print(f"\n{group}:")
        for module in modules:
            ok, detail = check_import(module)
            print_status(module, ok, detail if not ok else "")

    print("\nBaseline assets")
    for model_name, spec in BASELINE_ASSETS.items():
        print(f"\n{model_name}:")
        for rel_path in spec["required"]:
            path = root / rel_path
            print_status(rel_path, path.exists())
        for rel_path in spec["missing_by_default"]:
            path = root / rel_path
            print_status(rel_path, path.exists(), "expected after checkpoint download" if not path.exists() else "")

    print("\nGenerated outputs")
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        expected = len(manifest)
        for model_name in ("stargan-vc", "autovc", "ppg-vc", "diff-vc"):
            count = count_outputs(root, model_name, manifest)
            print(f"{model_name}: {count}/{expected}")
    else:
        print_status(str(manifest_path), False)


if __name__ == "__main__":
    main()
