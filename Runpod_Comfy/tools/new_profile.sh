#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <recipe-id>"
  echo "Example: $0 upscaler-v1"
  exit 1
fi

ID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPES="$ROOT/admin_recipes"
EXAMPLES="$RECIPES/examples"
TARGET="$RECIPES/$ID"

mkdir -p "$TARGET"

for kind in runpod nodes models; do
  src="$EXAMPLES/${kind}.example.yaml"
  dst="$TARGET/${kind}.yaml"
  if [[ -f "$dst" ]]; then
    echo "[skip] exists: $dst"
  else
    cp "$src" "$dst"
    sed -i "s/^profile_id: .*/profile_id: ${ID}/" "$dst" || true
    echo "[new]  $dst"
  fi
done

src_urls="$EXAMPLES/urls.example.txt"
dst_urls="$TARGET/urls.txt"
if [[ -f "$dst_urls" ]]; then
  echo "[skip] exists: $dst_urls"
else
  cp "$src_urls" "$dst_urls"
  echo "[new]  $dst_urls"
fi

echo
echo "Next:"
echo "  1) edit $TARGET/{runpod.yaml,nodes.yaml,models.yaml,urls.txt}"
echo "  2) Runpod_Comfy/tools/compile_profile.sh --id $ID"
