#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <profile-id>"
  echo "Example: $0 upscaler-v1"
  exit 1
fi

ID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$ROOT/profiles"

for kind in runpod nodes models; do
  src="$P/${kind}.example.yaml"
  dst="$P/${kind}.${ID}.yaml"
  if [[ -f "$dst" ]]; then
    echo "[skip] exists: $dst"
  else
    cp "$src" "$dst"
    sed -i "s/^profile_id: .*/profile_id: ${ID}/" "$dst" || true
    echo "[new] $dst"
  fi
done

u="$P/urls.${ID}.txt"
if [[ -f "$u" ]]; then
  echo "[skip] exists: $u"
else
  cp "$P/urls.txt" "$u"
  echo "[new] $u"
fi

echo "\nNext:"
echo "  1) edit profiles/*.${ID}.yaml"
echo "  2) Runpod_Comfy/tools/compile_profile.sh ${ID}"
