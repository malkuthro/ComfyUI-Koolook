#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <profile-id>"
  echo "Example: $0 example"
  exit 1
fi

PROFILE_ID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="$ROOT/profiles"
CONFIG_DIR="$ROOT/config"

RUNPOD_PROFILE="$PROFILES_DIR/runpod.${PROFILE_ID}.yaml"
NODES_PROFILE="$PROFILES_DIR/nodes.${PROFILE_ID}.yaml"
MODELS_PROFILE="$PROFILES_DIR/models.${PROFILE_ID}.yaml"
URLS_FILE="$PROFILES_DIR/urls.${PROFILE_ID}.txt"

if [[ ! -f "$RUNPOD_PROFILE" ]]; then
  echo "Missing profile: $RUNPOD_PROFILE"
  echo "Tip: copy from $PROFILES_DIR/runpod.example.yaml"
  exit 1
fi
if [[ ! -f "$NODES_PROFILE" ]]; then
  echo "Missing profile: $NODES_PROFILE"
  echo "Tip: copy from $PROFILES_DIR/nodes.example.yaml"
  exit 1
fi
if [[ ! -f "$MODELS_PROFILE" ]]; then
  echo "Missing profile: $MODELS_PROFILE"
  echo "Tip: copy from $PROFILES_DIR/models.example.yaml"
  exit 1
fi

mkdir -p "$CONFIG_DIR"

python3 - "$RUNPOD_PROFILE" "$NODES_PROFILE" "$MODELS_PROFILE" "$URLS_FILE" "$CONFIG_DIR" <<'PY'
import json
import pathlib
import re
import sys

runpod_path = pathlib.Path(sys.argv[1])
nodes_path = pathlib.Path(sys.argv[2])
models_path = pathlib.Path(sys.argv[3])
urls_path = pathlib.Path(sys.argv[4])
config_dir = pathlib.Path(sys.argv[5])


def parse_scalar(v: str):
    v = v.strip()
    if not v:
        return ""
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v)
        except ValueError:
            pass
    return v


def parse_simple_yaml(path: pathlib.Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    root = {}
    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if stripped.startswith("- "):
            raise ValueError(f"Top-level list not supported in {path}")

        if ":" not in raw:
            raise ValueError(f"Invalid line in {path}: {raw}")

        key, val = raw.split(":", 1)
        key = key.strip()
        val = val.strip()

        if val:
            root[key] = parse_scalar(val)
            i += 1
            continue

        # block value
        i += 1
        block = []
        while i < n:
            raw2 = lines[i]
            if not raw2.strip() or raw2.strip().startswith("#"):
                i += 1
                continue
            indent = len(raw2) - len(raw2.lstrip(" "))
            if indent < 2:
                break
            block.append(raw2)
            i += 1

        if block and block[0].lstrip().startswith("- "):
            items = []
            current = None
            for bl in block:
                indent = len(bl) - len(bl.lstrip(" "))
                s = bl.strip()
                if s.startswith("- "):
                    if current is not None:
                        items.append(current)
                    current = {}
                    payload = s[2:].strip()
                    if payload:
                        if ":" not in payload:
                            raise ValueError(f"Invalid list item in {path}: {bl}")
                        k, v = payload.split(":", 1)
                        current[k.strip()] = parse_scalar(v.strip())
                else:
                    if current is None:
                        raise ValueError(f"Malformed list block in {path}: {bl}")
                    if ":" not in s:
                        raise ValueError(f"Invalid nested mapping in {path}: {bl}")
                    k, v = s.split(":", 1)
                    current[k.strip()] = parse_scalar(v.strip())
            if current is not None:
                items.append(current)
            root[key] = items
        else:
            nested = {}
            for bl in block:
                s = bl.strip()
                if ":" not in s:
                    raise ValueError(f"Invalid nested mapping in {path}: {bl}")
                k, v = s.split(":", 1)
                nested[k.strip()] = parse_scalar(v.strip())
            root[key] = nested

    return root


def parse_urls(path: pathlib.Path):
    models = []
    if not path.exists():
        return models
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) < 2:
            raise ValueError(f"Invalid urls entry: {line}")
        url = parts[0]
        dest = parts[1]
        sha = parts[2] if len(parts) > 2 else ""
        enabled_raw = parts[3] if len(parts) > 3 else "true"
        name = parts[4] if len(parts) > 4 and parts[4] else pathlib.Path(dest).stem
        enabled = str(enabled_raw).lower() != "false"
        models.append({
            "name": name,
            "url": url,
            "dest": dest,
            "sha256": sha,
            "enabled": enabled,
        })
    return models

runpod = parse_simple_yaml(runpod_path)
nodes_data = parse_simple_yaml(nodes_path)
models_data = parse_simple_yaml(models_path)
url_models = parse_urls(urls_path)

base_image = runpod.get("base_image")
comfyui = runpod.get("comfyui", {}) if isinstance(runpod.get("comfyui"), dict) else {}
comfy_repo = comfyui.get("repo", "https://github.com/comfyanonymous/ComfyUI.git")
comfy_ref = comfyui.get("ref", "master")

if not base_image:
    raise ValueError(f"base_image is required in {runpod_path}")

nodes = nodes_data.get("nodes", [])
if not isinstance(nodes, list):
    raise ValueError(f"nodes must be a list in {nodes_path}")

for node in nodes:
    if "repo" not in node or "ref" not in node:
        raise ValueError(f"Each node requires repo and ref in {nodes_path}: {node}")
    if "target_dir" not in node or not node["target_dir"]:
        repo = str(node["repo"]).rstrip("/")
        node["target_dir"] = pathlib.Path(repo).name.replace(".git", "")
    node.setdefault("install", "[]")
    if isinstance(node["install"], str):
        node["install"] = []

models = models_data.get("models", [])
if not isinstance(models, list):
    raise ValueError(f"models must be a list in {models_path}")

for model in models:
    for req in ("name", "url", "dest"):
        if req not in model:
            raise ValueError(f"Model missing '{req}' in {models_path}: {model}")
    model.setdefault("sha256", "")
    model.setdefault("enabled", True)

models.extend(url_models)

comfy_lock = "\n".join([
    "# Generated by Runpod_Comfy/tools/compile_profile.sh",
    f"PROFILE_ID={runpod.get('profile_id', 'unknown')}",
    f"BASE_IMAGE={base_image}",
    f"COMFYUI_REPO={comfy_repo}",
    f"COMFYUI_REF={comfy_ref}",
    "",
])

(config_dir / "comfyui.lock").write_text(comfy_lock, encoding="utf-8")
(config_dir / "custom_nodes.lock.json").write_text(
    json.dumps({"version": 1, "nodes": nodes}, indent=2) + "\n",
    encoding="utf-8",
)
(config_dir / "models.json").write_text(
    json.dumps({"version": 1, "models": models}, indent=2) + "\n",
    encoding="utf-8",
)

print("=== Runpod_Comfy profile compile summary ===")
print(f"Profile ID    : {runpod.get('profile_id', 'unknown')}")
print(f"Base image    : {base_image}")
print(f"ComfyUI repo  : {comfy_repo}")
print(f"ComfyUI ref   : {comfy_ref}")
print(f"Custom nodes  : {len(nodes)}")
print(f"Models        : {len(models)} (includes urls.txt additions: {len(url_models)})")
print("Generated:")
print(f" - {config_dir / 'comfyui.lock'}")
print(f" - {config_dir / 'custom_nodes.lock.json'}")
print(f" - {config_dir / 'models.json'}")
PY

echo "[ok] compile complete"
