#!/usr/bin/env bash
# Apply local compatibility patches to freshly cloned baseline repositories.

set -euo pipefail

cd "$(dirname "$0")/.."

apply_patch_once() {
  local repo="$1"
  local patch="$2"
  local label="$3"

  if git -C "${repo}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
    echo "${label} patch already applied."
  else
    git -C "${repo}" apply --check "${patch}"
    git -C "${repo}" apply "${patch}"
    echo "Applied ${label} patch."
  fi
}

if [ -d repos/StarGAN-VC/.git ]; then
  apply_patch_once repos/StarGAN-VC ../../patches/stargan-vc-local-fixes.patch "StarGAN-VC"
else
  echo "Skipping StarGAN-VC patch: repos/StarGAN-VC is missing."
fi

if [ -d repos/ppg-vc/.git ]; then
  apply_patch_once repos/ppg-vc ../../patches/ppg-vc-device-option.patch "PPG-VC"
else
  echo "Skipping PPG-VC patch: repos/ppg-vc is missing."
fi
