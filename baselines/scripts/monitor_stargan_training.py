#!/usr/bin/env python3
"""Print StarGAN-VC feature extraction and training progress."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


EPOCH_RE = re.compile(r"epoch\s+(\d+),\s+mini-batch\s+(\d+)")


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def latest_log_line(log_path: Path) -> tuple[int | None, int | None, str]:
    if not log_path.exists():
        return None, None, "training log not found yet"

    latest = ""
    epoch = None
    batch = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_RE.search(line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                latest = line.strip()
    return epoch, batch, latest or "no epoch lines in log yet"


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def speaker_file_counts(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    return {
        speaker_dir.name: count_files(speaker_dir, "*.h5")
        for speaker_dir in sorted(root.iterdir())
        if speaker_dir.is_dir()
    }


def print_status(repo_dir: Path, dataset: str, experiment: str) -> None:
    dump_dir = repo_dir / "dump" / dataset
    feat_dir = dump_dir / "feat" / "train"
    norm_dir = dump_dir / "norm_feat" / "train"
    model_dir = repo_dir / "model" / dataset / experiment
    log_path = repo_dir / "logs" / dataset / experiment / f"train_{experiment}.log"

    config = load_config(model_dir)
    target_epochs = config.get("epochs", 2000)
    batch_size = config.get("BatchSize", 12)
    norm_counts = speaker_file_counts(norm_dir)
    batches_per_epoch = None
    if norm_counts:
        batches_per_epoch = max(1, min(norm_counts.values()) // max(1, int(batch_size)))

    epoch, batch, latest = latest_log_line(log_path)
    checkpoints = sorted(model_dir.glob("*.gen.pt"))

    print("=" * 72)
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"repo: {repo_dir}")
    print(f"experiment: {experiment}")
    print(f"raw feature files: {count_files(feat_dir, '*/*.h5')}")
    print(f"normalized feature files: {count_files(norm_dir, '*/*.h5')}")
    if norm_counts:
        print("normalized by speaker: " + ", ".join(f"{k}={v}" for k, v in norm_counts.items()))
    print(f"checkpoints: {len(checkpoints)}" + (f" latest={checkpoints[-1].name}" if checkpoints else ""))
    if epoch is not None:
        pct = 100.0 * epoch / float(target_epochs) if target_epochs else 0.0
        step_text = ""
        if batches_per_epoch:
            step_text = f" batch {batch}/{batches_per_epoch}"
        print(f"training: epoch {epoch}/{target_epochs}{step_text} ({pct:.2f}%)")
    else:
        print("training: not started or no logged minibatch yet")
    print(f"latest log: {latest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parents[1] / "repos" / "StarGAN-VC"
    parser.add_argument("--repo-dir", type=Path, default=default_repo)
    parser.add_argument("--dataset", default="arctic_4spk")
    parser.add_argument("--experiment", default="voicegrad_arctic_4spk")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        print_status(args.repo_dir.resolve(), args.dataset, args.experiment)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
