#!/usr/bin/env bash
# Clone the external baseline repositories expected by the pipeline.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p repos

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [ -d "${dest}/.git" ]; then
    echo "exists: ${dest}"
  else
    git clone "${url}" "${dest}"
  fi
}

clone_if_missing https://github.com/kamepong/StarGAN-VC repos/StarGAN-VC
clone_if_missing https://github.com/auspicious3000/autovc repos/autovc
clone_if_missing https://github.com/liusongxiang/ppg-vc repos/ppg-vc
clone_if_missing https://github.com/trinhtuanvubk/Diff-VC repos/Diff-VC

bash scripts/apply_baseline_patches.sh
