#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_ID=""
RECIPES_DIR="$ROOT/admin_recipes"
OUT_DIR=""
ACTIVATE=true
IMAGE_TAG_OVERRIDE=""

usage() {
  cat <<EOF
Usage:
  $0 --id <recipe-id> [--recipes-dir <dir>] [--out <dir>] [--image-tag <tag>] [--no-activate]
  $0 <recipe-id>   # backward-compatible

Defaults:
  recipes-dir: Runpod_Comfy/admin_recipes
  out:         Runpod_Comfy/builds/<recipe-id>
  activate:    true (copies generated files to Runpod_Comfy/config)
EOF
}

if [[ $# -eq 1 && "$1" != --* ]]; then
  RECIPE_ID="$1"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --id) RECIPE_ID="$2"; shift 2 ;;
      --recipes-dir) RECIPES_DIR="$2"; shift 2 ;;
      --out) OUT_DIR="$2"; shift 2 ;;
      --image-tag) IMAGE_TAG_OVERRIDE="$2"; shift 2 ;;
      --no-activate) ACTIVATE=false; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
  done
fi

[[ -z "$RECIPE_ID" ]] && { usage; exit 1; }
[[ -z "$OUT_DIR" ]] && OUT_DIR="$ROOT/builds/$RECIPE_ID"

RECIPE_DIR="$RECIPES_DIR/$RECIPE_ID"
RUNPOD_PROFILE="$RECIPE_DIR/runpod.yaml"
NODES_PROFILE="$RECIPE_DIR/nodes.yaml"
MODELS_PROFILE="$RECIPE_DIR/models.yaml"
URLS_FILE="$RECIPE_DIR/urls.txt"
CONFIG_DIR="$ROOT/config"

for f in "$RUNPOD_PROFILE" "$NODES_PROFILE" "$MODELS_PROFILE"; do
  [[ -f "$f" ]] || { echo "Missing: $f"; exit 1; }
done
mkdir -p "$OUT_DIR"

python3 - "$RUNPOD_PROFILE" "$NODES_PROFILE" "$MODELS_PROFILE" "$URLS_FILE" "$OUT_DIR" "$RECIPE_ID" "$IMAGE_TAG_OVERRIDE" <<'PY'
import json, pathlib, re, sys
runpod_path = pathlib.Path(sys.argv[1]); nodes_path = pathlib.Path(sys.argv[2]); models_path = pathlib.Path(sys.argv[3]); urls_path = pathlib.Path(sys.argv[4]); out_dir = pathlib.Path(sys.argv[5]); recipe_id = sys.argv[6]; image_tag_override = sys.argv[7]

def scalar(v):
    v=v.strip()
    if not v: return ""
    if v.lower()=="true": return True
    if v.lower()=="false": return False
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")): return v[1:-1]
    if re.fullmatch(r"-?\d+", v): return int(v)
    return v

def parse(path):
    lines=path.read_text(encoding='utf-8').splitlines(); root={}; i=0
    while i<len(lines):
        raw=lines[i]; s=raw.strip(); i+=1
        if not s or s.startswith('#'): continue
        if ':' not in raw: raise ValueError(f"Invalid line: {raw}")
        k,v=raw.split(':',1); k=k.strip(); v=v.strip()
        if v:
            root[k]=scalar(v); continue
        block=[]
        while i<len(lines):
            r=lines[i]; st=r.strip()
            if not st or st.startswith('#'): i+=1; continue
            if (len(r)-len(r.lstrip(' ')))<2: break
            block.append(r); i+=1
        if block and block[0].lstrip().startswith('- '):
            items=[]; cur=None
            for b in block:
                st=b.strip()
                if st.startswith('- '):
                    if cur is not None: items.append(cur)
                    cur={}; payload=st[2:].strip()
                    if payload:
                        kk,vv=payload.split(':',1); cur[kk.strip()]=scalar(vv.strip())
                else:
                    kk,vv=st.split(':',1); cur[kk.strip()]=scalar(vv.strip())
            if cur is not None: items.append(cur)
            root[k]=items
        else:
            d={}
            for b in block:
                kk,vv=b.strip().split(':',1); d[kk.strip()]=scalar(vv.strip())
            root[k]=d
    return root

def parse_urls(path):
    if not path.exists(): return []
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        s=line.strip()
        if not s or s.startswith('#'): continue
        p=[x.strip() for x in s.split('|')]
        if len(p)<2: raise ValueError(f"Bad urls entry: {line}")
        url,dest=p[0],p[1]; sha=p[2] if len(p)>2 else ""; enabled=(p[3].lower()!='false') if len(p)>3 and p[3] else True; name=p[4] if len(p)>4 and p[4] else pathlib.Path(dest).stem
        out.append({"name":name,"url":url,"dest":dest,"sha256":sha,"enabled":enabled})
    return out

runpod=parse(runpod_path); nodes_data=parse(nodes_path); models_data=parse(models_path)
base_image=runpod.get('base_image'); comfy=runpod.get('comfyui',{}) if isinstance(runpod.get('comfyui'),dict) else {}
if not base_image: raise ValueError('base_image is required')
image_name=runpod.get('image_name','')
image_tag= image_tag_override or runpod.get('image_tag', recipe_id)
nodes=nodes_data.get('nodes',[]); models=models_data.get('models',[])
url_models=parse_urls(urls_path)
for n in nodes:
    if 'repo' not in n or 'ref' not in n: raise ValueError(f'node requires repo/ref: {n}')
    n.setdefault('target_dir', pathlib.Path(str(n['repo']).rstrip('/')).name.replace('.git',''))
    if n.get('install') in ('',None): n['install']=[]
for m in models:
    for req in ('name','url','dest'):
      if req not in m: raise ValueError(f'model missing {req}: {m}')
    m.setdefault('sha256',''); m.setdefault('enabled',True)
models.extend(url_models)

comfy_lock='\n'.join([
    '# Generated by Runpod_Comfy/tools/compile_profile.sh',
    f'RECIPE_ID={recipe_id}',
    f'BASE_IMAGE={base_image}',
    f'IMAGE_NAME={image_name}',
    f'RUNPOD_IMAGE_TAG={image_tag}',
    f"COMFYUI_REPO={comfy.get('repo','https://github.com/comfyanonymous/ComfyUI.git')}",
    f"COMFYUI_REF={comfy.get('ref','master')}",
    ''
])
(out_dir/'comfyui.lock').write_text(comfy_lock, encoding='utf-8')
(out_dir/'custom_nodes.lock.json').write_text(json.dumps({'version':1,'nodes':nodes},indent=2)+'\n',encoding='utf-8')
(out_dir/'models.json').write_text(json.dumps({'version':1,'models':models},indent=2)+'\n',encoding='utf-8')
print('=== compile summary ===')
print(f'Recipe ID     : {recipe_id}')
print(f'Base image    : {base_image}')
print(f'Image name    : {image_name or "(unset)"}')
print(f'Image tag     : {image_tag}')
print(f'ComfyUI ref   : {comfy.get("ref","master")}')
print(f'Custom nodes  : {len(nodes)}')
print(f'Models        : {len(models)} (urls.txt additions: {len(url_models)})')
print(f'Output dir    : {out_dir}')
PY

if [[ "$ACTIVATE" == true ]]; then
  mkdir -p "$CONFIG_DIR"
  cp "$OUT_DIR/comfyui.lock" "$CONFIG_DIR/comfyui.lock"
  cp "$OUT_DIR/custom_nodes.lock.json" "$CONFIG_DIR/custom_nodes.lock.json"
  cp "$OUT_DIR/models.json" "$CONFIG_DIR/models.json"
  echo "Activated build into: $CONFIG_DIR"
fi

echo "[ok] compile complete"
