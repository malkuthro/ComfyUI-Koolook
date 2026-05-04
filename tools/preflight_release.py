#!/usr/bin/env python3
"""
Pre-flight release validation for ComfyUI-Koolook.

Runs four checks and exits non-zero on any failure:

  1. static       AST extraction of NODE_CLASS_MAPPINGS keys; verifies all
                  *.py files parse and that we can produce a definitive list
                  of registered node IDs without importing torch.
  2. dispatch     Stub-VAE round through the v2_3_3 wrapper; verifies all
                  five rank/VAE dispatch branches still wire correctly.
  3. manager-meta Fetches ComfyUI-Manager's extension-node-map.json upstream
                  and diffs the entry for our repo against the AST-extracted
                  NODE_CLASS_MAPPINGS. Catches phantom-nodes / missing-nodes
                  drift (the issue #44 class of bug).
  4. workflows    Walks tests/workflows/*.json, extracts node IDs that look
                  like Koolook-pack IDs, verifies each one is currently
                  registered (i.e. wouldn't break a saved workflow).

Usage:
    python tools/preflight_release.py                # run all checks
    python tools/preflight_release.py --check static
    python tools/preflight_release.py --check workflows --verbose
    python tools/preflight_release.py --skip manager-meta  # offline mode

Exit codes:
    0  all run checks passed
    1  one or more checks failed
    2  invalid CLI usage
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/malkuthro/ComfyUI-Koolook"
MANAGER_EXT_MAP_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/"
    "main/extension-node-map.json"
)


# ANSI colors for terminal output (no-op when piped or NO_COLOR set).
def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def _green(s: str) -> str: return _color(s, "32")
def _red(s: str) -> str: return _color(s, "31")
def _yellow(s: str) -> str: return _color(s, "33")
def _cyan(s: str) -> str: return _color(s, "36")
def _dim(s: str) -> str: return _color(s, "2")


# ───────────────────────────────────────────────────────────────────────
# Check 1 — static (AST extraction)
# ───────────────────────────────────────────────────────────────────────


# Files / folders to skip when walking for *.py.
_AST_SKIP_DIRS = {
    "__pycache__", ".venv", ".git", "upscaler_FIX",
    "nuke_CAM_exporter", "node_modules",
}


def _iter_python_files() -> Iterable[Path]:
    for p in REPO_ROOT.rglob("*.py"):
        if any(part in _AST_SKIP_DIRS for part in p.parts):
            continue
        yield p


def _extract_literal_dict_keys(node: ast.AST) -> set[str]:
    """Pull literal string keys out of an ast.Dict node, ignoring computed keys."""
    if not isinstance(node, ast.Dict):
        return set()
    keys = set()
    for k in node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.add(k.value)
    return keys


def check_static(verbose: bool = False) -> tuple[bool, set[str], list[str]]:
    """
    Walk all .py files, AST-parse each, collect every literal-dict
    `NODE_CLASS_MAPPINGS = {...}` and every literal `SKIP_VERSION_SUFFIX = {...}`
    set, and return the union of source IDs plus a list of any parse errors.

    Caveats: doesn't follow runtime-computed mappings (e.g. dict comprehensions
    or function calls that produce mappings). For the current Radiance fork,
    SKIP_VERSION_SUFFIX makes the user-facing IDs match the literal source-dict
    keys exactly, so AST extraction matches reality. Future fork additions that
    rely on suffix mangling will need this check extended.
    """
    print(_cyan("[1/4] static — AST extraction of NODE_CLASS_MAPPINGS"))
    parse_errors: list[str] = []
    all_node_ids: set[str] = set()
    files_seen = 0

    for py_path in _iter_python_files():
        files_seen += 1
        try:
            source = py_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append(f"{py_path.relative_to(REPO_ROOT)}: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "NODE_CLASS_MAPPINGS"
                    ):
                        keys = _extract_literal_dict_keys(node.value)
                        all_node_ids.update(keys)
                        if verbose and keys:
                            rel = py_path.relative_to(REPO_ROOT)
                            print(_dim(f"  {rel}: {sorted(keys)}"))

    if parse_errors:
        print(_red(f"  FAIL — {len(parse_errors)} parse errors:"))
        for err in parse_errors:
            print(_red(f"    {err}"))
        return False, all_node_ids, parse_errors

    if not all_node_ids:
        print(_red("  FAIL — no NODE_CLASS_MAPPINGS literal dicts found anywhere"))
        return False, all_node_ids, ["no NODE_CLASS_MAPPINGS found"]

    print(
        _green(f"  PASS — {files_seen} *.py files parsed, "
               f"{len(all_node_ids)} unique node IDs collected:")
    )
    for nid in sorted(all_node_ids):
        print(_dim(f"    {nid}"))
    return True, all_node_ids, []


# ───────────────────────────────────────────────────────────────────────
# Check 2 — dispatch (stub-VAE roundtrip)
# ───────────────────────────────────────────────────────────────────────


def check_dispatch(verbose: bool = False) -> bool:
    """
    Run the v2_3_3 VAE wrapper through every rank/VAE-type combination using a
    FakeTensor stub. No torch dependency — the stub mimics just enough of
    torch.Tensor to drive the dispatch logic.
    """
    print(_cyan("[2/4] dispatch — VAE rank/VAE-type branches"))
    import types

    class FT:
        def __init__(self, shape):
            self.shape = tuple(shape)
            self.dtype = "float32"
            self.device = "cpu"
        @property
        def ndim(self): return len(self.shape)
        @property
        def T(self):
            s = list(self.shape)
            if len(s) >= 2:
                s[-1], s[-2] = s[-2], s[-1]
            return FT(s)
        def clone(self): return FT(self.shape)
        def float(self): return self
        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 3 and key[2] is Ellipsis:
                return FT((self.shape[0],) + self.shape[2:])
            if isinstance(key, tuple) and len(key) == 4 and key[3] is Ellipsis:
                return FT((self.shape[0], self.shape[1]) + self.shape[3:])
            if isinstance(key, tuple) and key[0] is Ellipsis:
                return FT(self.shape[:-1] + (3,))
            return self
        def reshape(self, *a): return FT(a)
        def to(self, **k): return self
        def __mul__(self, o): return self
        def __rmul__(self, o): return self
        def __pow__(self, o): return self
        def __sub__(self, o): return self
        def clamp(self, *a, **k): return self
        def __matmul__(self, o):
            return FT(self.shape[:-1] + (o.shape[-1],))

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = FT
    torch_stub.stack = lambda ts, dim=0: (
        lambda s: (s.insert(dim, len(ts)), FT(s))[1]
    )(list(ts[0].shape))
    torch_stub.cat = lambda ts, dim=0: (
        lambda s: (s.__setitem__(dim, sum(x.shape[dim] for x in ts)), FT(s))[1]
    )(list(ts[0].shape))
    torch_stub.where = lambda c, a, b: a
    torch_stub.sign = lambda x: x
    torch_stub.abs = lambda x: x
    torch_stub.pow = lambda x, p: x
    torch_stub.tanh = lambda x: x
    torch_stub.clamp = lambda x, *a, **k: x
    torch_stub.tensor = lambda *a, **k: FT((3, 3))
    torch_stub.float32 = "float32"
    torch_stub.zeros = lambda shape, dtype=None, device=None: FT(shape)
    nn_F = types.ModuleType("torch.nn.functional")
    nn_F.pad = lambda x, *a, **k: x
    nn_F.avg_pool2d = lambda x, *a, **k: x
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.functional = nn_F
    torch_stub.nn = nn_mod

    # Track every sys.modules key we add so the finally block at the end can
    # pop them cleanly even if any of the exec_module / constructor calls below
    # raise. Without this, a partial-load failure would leave the torch stub
    # and the _pf_pkg* entries shadowing real modules for the rest of the run.
    injected_modules: list[str] = []

    import importlib.util
    import importlib.machinery
    pkg_path = REPO_ROOT / "forks" / "radiance_koolook" / "versions" / "v2_3_3"

    try:
        sys.modules["torch"] = torch_stub
        injected_modules.append("torch")
        sys.modules["torch.nn"] = nn_mod
        injected_modules.append("torch.nn")
        sys.modules["torch.nn.functional"] = nn_F
        injected_modules.append("torch.nn.functional")

        pkg_spec = importlib.machinery.ModuleSpec(
            name="_pf_pkg", loader=None, is_package=True
        )
        pkg_spec.submodule_search_locations = [str(pkg_path)]
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules["_pf_pkg"] = pkg
        injected_modules.append("_pf_pkg")

        ch_spec = importlib.util.spec_from_file_location(
            "_pf_pkg.color_helpers", str(pkg_path / "color_helpers.py")
        )
        ch_mod = importlib.util.module_from_spec(ch_spec)
        sys.modules["_pf_pkg.color_helpers"] = ch_mod
        injected_modules.append("_pf_pkg.color_helpers")
        ch_spec.loader.exec_module(ch_mod)

        nv_spec = importlib.util.spec_from_file_location(
            "_pf_pkg.nodes_vae", str(pkg_path / "nodes_vae.py")
        )
        nv_mod = importlib.util.module_from_spec(nv_spec)
        sys.modules["_pf_pkg.nodes_vae"] = nv_mod
        injected_modules.append("_pf_pkg.nodes_vae")
        nv_spec.loader.exec_module(nv_mod)

        return _check_dispatch_run_cases(nv_mod, FT, verbose)
    finally:
        # Always pop every module we injected. Catches both the success path
        # and any exception during exec_module / constructor / case execution.
        for mod in injected_modules:
            sys.modules.pop(mod, None)


def _check_dispatch_run_cases(nv_mod, FT, verbose: bool) -> bool:
    """The actual five test cases, factored out so the try/finally above can
    wrap module injection cleanly. Receives the loaded `nodes_vae` module and
    the `FakeTensor`-equivalent class via parameters."""
    class V2D:
        def encode(self, p):
            return FT((1, 16, p.shape[1] // 8, p.shape[2] // 8))
        def decode(self, lat):
            return FT((lat.shape[0], lat.shape[2] * 8, lat.shape[3] * 8, 3))

    class V3D:
        latent_dim = 3
        def encode(self, p):
            if p.ndim == 5:
                return FT(
                    (1, 16, p.shape[1] // 4, p.shape[2] // 8, p.shape[3] // 8)
                )
            return FT((1, 16, p.shape[1] // 8, p.shape[2] // 8))
        def decode(self, lat):
            if lat.ndim == 5:
                return FT(
                    (1, lat.shape[2] * 4, lat.shape[3] * 8, lat.shape[4] * 8, 3)
                )
            return FT((lat.shape[0], lat.shape[2] * 8, lat.shape[3] * 8, 3))

    enc = nv_mod.Easy_hdr_VAE_encode()
    dec = nv_mod.Easy_hdr_VAE_decode()
    common_enc = dict(
        source_space="Raw", hdr_mode="Passthrough",
        exposure=0.0, alpha_handling="Preserve", latent_sampling="sample",
    )
    common_dec = dict(
        target_space="Raw", exposure_adjust=0.0,
        hdr_mode="Clip (SDR)", source_space="Linear",
    )

    cases: list[tuple[str, callable]] = []

    def case_4d_image():
        lat, dbg, _ = enc.encode(FT((2, 64, 64, 3)), V2D(), **common_enc)
        assert lat["samples"].ndim == 4 and "path=image" in dbg

    def case_5d_2dvae():
        lat, dbg, _ = enc.encode(FT((1, 8, 64, 64, 3)), V2D(), **common_enc)
        assert (
            lat["samples"].ndim == 5
            and lat["samples"].shape[2] == 8
            and "path=video-iter" in dbg
        )

    def case_5d_3dvae():
        lat, dbg, _ = enc.encode(FT((1, 8, 64, 64, 3)), V3D(), **common_enc)
        assert "path=video-3d" in dbg

    def case_5d_lat_decode():
        img, dbg = dec.decode({"samples": FT((1, 16, 8, 8, 8))}, V2D(), **common_dec)
        assert img.ndim == 4 and img.shape[0] == 8 and "path=video-iter" in dbg

    def case_4d_lat_decode():
        img, dbg = dec.decode({"samples": FT((2, 16, 8, 8))}, V2D(), **common_dec)
        assert "path=image" in dbg

    cases = [
        ("4D image, 2D-VAE", case_4d_image),
        ("5D + 2D-VAE iter", case_5d_2dvae),
        ("5D + 3D-VAE pass", case_5d_3dvae),
        ("5D latent decode iter", case_5d_lat_decode),
        ("4D latent decode pass", case_4d_lat_decode),
    ]

    failures: list[str] = []
    for name, fn in cases:
        try:
            fn()
            if verbose:
                print(_dim(f"    OK {name}"))
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    # NOTE: sys.modules cleanup is handled by the try/finally in the parent
    # check_dispatch() function, so we don't need to pop anything here.

    if failures:
        print(_red(f"  FAIL — {len(failures)} dispatch branch(es) broken:"))
        for f in failures:
            print(_red(f"    {f}"))
        return False
    print(_green(f"  PASS — all {len(cases)} dispatch branches OK"))
    return True


# ───────────────────────────────────────────────────────────────────────
# Check 3 — manager-meta (extension-node-map drift)
# ───────────────────────────────────────────────────────────────────────


def check_manager_meta(actual_ids: set[str], verbose: bool = False) -> bool:
    """
    Compare ComfyUI-Manager's upstream extension-node-map entry for our repo
    against the AST-extracted node IDs. Reports phantom IDs (in upstream but
    not in our code) and missing IDs (in our code but not in upstream).
    """
    print(_cyan("[3/4] manager-meta — extension-node-map.json drift"))
    if not MANAGER_EXT_MAP_URL.startswith("https://"):
        # Defensive: hardcoded URL above is https; this guard ensures any
        # future edit can't accidentally introduce a non-https scheme that
        # would let urlopen pick up file:// or other unsafe schemes.
        print(_yellow("  SKIP — manager-meta URL is not https"))
        return True
    try:
        # Bandit B310: urlopen scheme check is enforced by the https guard
        # one line above; the URL is hardcoded to a github.com raw path.
        with urllib.request.urlopen(MANAGER_EXT_MAP_URL, timeout=20) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        print(_yellow(f"  SKIP — could not fetch upstream map: {exc}"))
        return True

    entry = data.get(REPO_URL)
    if entry is None:
        print(_yellow(f"  SKIP — no entry for {REPO_URL} in upstream map yet"))
        return True

    upstream_ids = set(entry[0]) if isinstance(entry, list) and entry else set()

    phantom = upstream_ids - actual_ids
    missing = actual_ids - upstream_ids

    if phantom or missing:
        print(_red("  FAIL — drift between code and Manager metadata:"))
        if phantom:
            print(_red(f"    Manager lists {len(phantom)} ID(s) not in our code:"))
            for x in sorted(phantom):
                print(_red(f"      - {x}"))
        if missing:
            print(_red(f"    Our code registers {len(missing)} ID(s) not in Manager:"))
            for x in sorted(missing):
                print(_red(f"      - {x}"))
        print(
            _yellow(
                "  -> Manager metadata is community-maintained. "
                "Drift here means an issue or PR to ltdrdata/ComfyUI-Manager. "
                "See issue #44 for the canonical write-up."
            )
        )
        return False

    print(_green(f"  PASS — Manager metadata matches our {len(actual_ids)} IDs"))
    return True


# ───────────────────────────────────────────────────────────────────────
# Check 4 — workflows (fixture node-ID stability)
# ───────────────────────────────────────────────────────────────────────


# Heuristic: a node `type` value in a workflow JSON is treated as a Koolook
# pack ID if it matches any of these patterns.
_KOOLOOK_ID_PATTERNS = [
    re.compile(r"(?i)koolook"),         # contains "koolook" anywhere
    re.compile(r"^Easy[A-Z_]"),         # starts with "Easy" + capital/underscore
    re.compile(r"^easy_[A-Za-z]"),      # starts with "easy_" + any letter
                                        # (covers easy_ImageBatch with capital I)
]


def _looks_like_koolook_id(nid: str) -> bool:
    return any(p.search(nid) for p in _KOOLOOK_ID_PATTERNS)


def _extract_workflow_node_types(data: Any) -> set[str]:
    """
    Handles both ComfyUI canvas format (top-level dict with `nodes` array)
    and API format (top-level dict where each value has `class_type`).
    """
    types: set[str] = set()
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        for n in data["nodes"]:
            t = n.get("type")
            if isinstance(t, str):
                types.add(t)
    elif isinstance(data, dict):
        for _, v in data.items():
            if isinstance(v, dict) and isinstance(v.get("class_type"), str):
                types.add(v["class_type"])
    return types


def check_workflows(actual_ids: set[str], verbose: bool = False) -> bool:
    print(_cyan("[4/4] workflows — fixture node-ID stability"))
    fixtures_dir = REPO_ROOT / "tests" / "workflows"
    if not fixtures_dir.is_dir():
        print(_yellow("  SKIP — no tests/workflows/ folder yet"))
        return True

    fixtures = sorted(fixtures_dir.glob("*.json"))
    if not fixtures:
        print(_yellow("  SKIP — no *.json fixtures present"))
        return True

    failures: list[str] = []
    for fx in fixtures:
        try:
            data = json.loads(fx.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{fx.name}: invalid JSON ({exc})")
            continue

        types = _extract_workflow_node_types(data)
        koolook_types = {t for t in types if _looks_like_koolook_id(t)}
        missing = koolook_types - actual_ids

        if verbose:
            print(_dim(
                f"  {fx.name}: {len(types)} types, "
                f"{len(koolook_types)} look like Koolook, "
                f"{len(missing)} missing"
            ))
        if missing:
            for nid in sorted(missing):
                failures.append(f"{fx.name}: {nid} (no longer registered)")

    if failures:
        print(_red(f"  FAIL — {len(failures)} workflow regression(s):"))
        for f in failures:
            print(_red(f"    {f}"))
        print(
            _yellow(
                "  -> Either add a back-compat alias for the missing ID, "
                "or update the fixture if the rename was intentional + announced."
            )
        )
        return False

    total_koolook_refs = sum(
        len({t for t in _extract_workflow_node_types(json.loads(fx.read_text("utf-8")))
             if _looks_like_koolook_id(t)})
        for fx in fixtures
    )
    print(_green(
        f"  PASS — {len(fixtures)} fixture(s) reference "
        f"{total_koolook_refs} Koolook node ID(s), all currently registered"
    ))
    return True


# ───────────────────────────────────────────────────────────────────────
# Orchestration
# ───────────────────────────────────────────────────────────────────────


CHECKS = ("static", "dispatch", "manager-meta", "workflows")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", choices=CHECKS, action="append",
        help="Run only the named check (repeatable). Default: run all.",
    )
    parser.add_argument(
        "--skip", choices=CHECKS, action="append",
        help="Skip the named check (repeatable).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-check details.",
    )
    args = parser.parse_args()

    selected = set(args.check or CHECKS)
    skipped = set(args.skip or ())
    to_run = [c for c in CHECKS if c in selected and c not in skipped]
    if not to_run:
        print(_red("No checks selected after applying --check / --skip"), file=sys.stderr)
        return 2

    print(_cyan(f"Pre-flight running: {', '.join(to_run)}"))
    print()

    actual_ids: set[str] = set()
    results: dict[str, bool] = {}

    if "static" in to_run:
        ok, actual_ids, _ = check_static(verbose=args.verbose)
        results["static"] = ok
        print()

    if "dispatch" in to_run:
        results["dispatch"] = check_dispatch(verbose=args.verbose)
        print()

    if "manager-meta" in to_run:
        if not actual_ids and "static" not in to_run:
            ok, actual_ids, _ = check_static(verbose=False)
        results["manager-meta"] = check_manager_meta(actual_ids, verbose=args.verbose)
        print()

    if "workflows" in to_run:
        if not actual_ids and "static" not in to_run:
            ok, actual_ids, _ = check_static(verbose=False)
        results["workflows"] = check_workflows(actual_ids, verbose=args.verbose)
        print()

    failed = [n for n, ok in results.items() if not ok]

    print(_cyan("Summary"))
    for n in to_run:
        marker = _green("PASS") if results.get(n) else _red("FAIL")
        print(f"  {marker}  {n}")

    if failed:
        print()
        print(_red(f"{len(failed)} check(s) failed: {', '.join(failed)}"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
