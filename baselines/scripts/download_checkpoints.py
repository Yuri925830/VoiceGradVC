#!/usr/bin/env python3
"""Download baseline checkpoints that have Google Drive IDs in the repo READMEs.

This intentionally does not cover AutoVC's IBM Box links; those are less stable
for scripted downloads and should be downloaded manually from the AutoVC README.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DIFF_VC_FILES = {
    "spk_encoder": {
        "id": "1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1",
        "path": "repos/Diff-VC/checkpts/spk_encoder/pretrained.pt",
    },
    "vocoder_generator": {
        "id": "10khlrM645pTbQ4rc2aNEYPba8RFDBkW-",
        "path": "repos/Diff-VC/checkpts/vocoder/generator",
    },
    "vc_libritts": {
        "id": "18Xbme0CTVo58p2vOHoTQm8PBGW7oEjAy",
        "path": "repos/Diff-VC/checkpts/vc/vc_libritts_wodyn.pt",
    },
    "vc_vctk": {
        "id": "12s9RPmwp9suleMkBCVetD8pub7wsDAy4",
        "path": "repos/Diff-VC/checkpts/vc/vc_vctk_wodyn.pt",
    },
}

PPG_FOLDER_ID = "1JeFntg2ax9gX4POFbQwcS85eC9hyQ6W6"
STARGAN_HIFIGAN_FOLDER_ID = "1RvagKsKaCih0qhRP6XkSF07r3uNFhB5T"
STARGAN_PWG_FOLDER_ID = "1zRYZ9dx16dONn1SEuO4wXjjgJHaYSKwb"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def gdown_file(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"exists: {output}")
        return
    run(["gdown", file_id, "-O", str(output)])


def gdown_folder(folder_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run(["gdown", "--folder", f"https://drive.google.com/drive/folders/{folder_id}", "-O", str(output_dir)])


def copy_resemblyzer_speaker_encoder(output: Path) -> None:
    if output.exists():
        print(f"exists: {output}")
        return
    try:
        import resemblyzer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Diff-VC speaker encoder fallback needs resemblyzer. Install it with:\n"
            "  conda run -n voicegrad-baselines python -m pip install resemblyzer"
        ) from exc

    source = Path(resemblyzer.__file__).resolve().parent / "pretrained.pt"
    if not source.exists():
        raise SystemExit(f"Resemblyzer is installed, but pretrained.pt was not found at {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    print(f"copied: {source} -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    baselines_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--baselines-root", type=Path, default=baselines_root)
    parser.add_argument(
        "target",
        choices=["diff-vc", "ppg-vc", "stargan-hifigan", "stargan-pwg"],
        help="Checkpoint group to download.",
    )
    parser.add_argument(
        "--diff-vc-model",
        choices=["libritts", "vctk", "both"],
        default="libritts",
        help="Which Diff-VC conversion checkpoint to download.",
    )
    args = parser.parse_args()
    root = args.baselines_root.resolve()

    if args.target == "diff-vc":
        copy_resemblyzer_speaker_encoder(root / DIFF_VC_FILES["spk_encoder"]["path"])
        keys = ["vocoder_generator"]
        if args.diff_vc_model in ("libritts", "both"):
            keys.append("vc_libritts")
        if args.diff_vc_model in ("vctk", "both"):
            keys.append("vc_vctk")
        for key in keys:
            spec = DIFF_VC_FILES[key]
            gdown_file(spec["id"], root / spec["path"])
    elif args.target == "ppg-vc":
        gdown_folder(PPG_FOLDER_ID, root / "downloads" / "ppg-vc")
        print("Downloaded PPG-VC folder to baselines/downloads/ppg-vc.")
        print("Move/copy the selected PPG2Mel checkpoint into repos/ppg-vc/checkpts/...")
    elif args.target == "stargan-hifigan":
        gdown_folder(STARGAN_HIFIGAN_FOLDER_ID, root / "downloads" / "stargan-hifigan")
        print("Downloaded StarGAN HiFi-GAN assets to baselines/downloads/stargan-hifigan.")
    elif args.target == "stargan-pwg":
        gdown_folder(STARGAN_PWG_FOLDER_ID, root / "downloads" / "stargan-pwg")
        print("Downloaded StarGAN PWG assets to baselines/downloads/stargan-pwg.")


if __name__ == "__main__":
    main()
