#!/usr/bin/env python3
"""
Pre-flight release validation for ComfyUI-Koolook.

Runs the release-gate checks and exits non-zero on any failure:

  1. static       AST extraction of NODE_CLASS_MAPPINGS keys; verifies all
                  *.py files parse and that we can produce a definitive list
                  of registered node IDs without importing torch.
  2. dispatch     Stub-VAE round through the v2_3_3 wrapper; verifies all
                  five rank/VAE dispatch branches still wire correctly.
  3. workflows    Walks tests/workflows/*.json, extracts node IDs that look
                  like Koolook-pack IDs, verifies each one is currently
                  registered (i.e. wouldn't break a saved workflow).

Optional advisory check:

  manager-meta    Fetches ComfyUI-Manager's extension-node-map.json upstream
                  and diffs the entry for our repo against the AST-extracted
                  NODE_CLASS_MAPPINGS. This catches phantom-nodes /
                  missing-nodes drift, but does not block releases because
                  upstream Manager metadata is maintained independently.

Usage:
    python tools/preflight_release.py                # run release gates
    python tools/preflight_release.py --check static
    python tools/preflight_release.py --check workflows --verbose
    python tools/preflight_release.py --check manager-meta  # advisory only

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


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuses to follow a 30x redirect to a non-https target.

    `urllib.request.urlopen` follows redirects by default; a one-time
    https URL check at the call site doesn't protect against an
    intermediate hop redirecting to file:// or http:// (Bandit B310's
    real concern). This handler re-validates each redirect target.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise urllib.error.URLError(
                f"refusing redirect to non-https scheme: {newurl}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# ANSI escape codes (https://en.wikipedia.org/wiki/ANSI_escape_code).
# No-op when piped, redirected, or NO_COLOR is set.
_ANSI = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "dim": "2",
}


def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def _green(s: str) -> str: return _color(s, _ANSI["green"])
def _red(s: str) -> str: return _color(s, _ANSI["red"])
def _yellow(s: str) -> str: return _color(s, _ANSI["yellow"])
def _cyan(s: str) -> str: return _color(s, _ANSI["cyan"])
def _dim(s: str) -> str: return _color(s, _ANSI["dim"])


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


def check_static(verbose: bool = False) -> tuple[bool, set[str]]:
    """
    Walk all .py files, AST-parse each, and collect every literal-dict
    `NODE_CLASS_MAPPINGS = {...}` source-side key. Returns (ok, ids).

    What this DOES NOT do (and the workflow check is the safety net):
      - Follow runtime-computed mappings (dict comprehensions, function
        calls like `_namespace_mappings(...)` in the v2_3_3 wrapper).
      - Apply `NAMESPACE_SUFFIX` / `SKIP_VERSION_SUFFIX` logic — the
        current code only walks `NODE_CLASS_MAPPINGS`, not the fork's
        suffix-mangling glue.

    For the Radiance Koolook v2_3_3 fork, this happens to work because
    every exposed VAE ID is in `SKIP_VERSION_SUFFIX`, so the source-dict
    keys match the user-facing IDs. The first time a non-skipped ID lands
    in a fork, the workflow-fixture check (4) will surface the mismatch
    by reporting the suffixed live ID as "missing" — at that point this
    function should grow SKIP_VERSION_SUFFIX extraction.
    """
    print(_cyan("static — AST extraction of NODE_CLASS_MAPPINGS"))
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
        return False, all_node_ids

    if not all_node_ids:
        print(_red("  FAIL — no NODE_CLASS_MAPPINGS literal dicts found anywhere"))
        return False, all_node_ids

    print(
        _green(f"  PASS — {files_seen} *.py files parsed, "
               f"{len(all_node_ids)} unique node IDs collected"
               + (":" if verbose else ""))
    )
    if verbose:
        for nid in sorted(all_node_ids):
            print(_dim(f"    {nid}"))
    return True, all_node_ids


# ───────────────────────────────────────────────────────────────────────
# Check 2 — dispatch (stub-VAE roundtrip)
# ───────────────────────────────────────────────────────────────────────


def check_dispatch(verbose: bool = False) -> bool:
    """
    Run the v2_3_3 VAE wrapper through every rank/VAE-type combination using
    a FakeTensor stub. No torch dependency — the stub mimics just enough of
    `torch.Tensor` to drive the dispatch logic.

    Implementation lives in `tools/_preflight_dispatch.py` (extracted to keep
    this orchestrator under the project's ~400-line file-size guideline).
    """
    print(_cyan("dispatch — VAE rank/VAE-type branches"))

    # Lazy import: the sibling module shares the same directory as this file
    # but isn't importable as `tools._preflight_dispatch` unless tools/ is on
    # sys.path. Load it directly via importlib so working dir doesn't matter.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_preflight_dispatch",
        Path(__file__).resolve().parent / "_preflight_dispatch.py",
    )
    dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch)

    success, failures = dispatch.run(verbose=verbose, repo_root=REPO_ROOT)

    if failures:
        print(_red(f"  FAIL — {len(failures)} dispatch branch(es) broken:"))
        for f in failures:
            print(_red(f"    {f}"))
        return False
    print(_green("  PASS — all 5 dispatch branches OK"))
    return success


# ───────────────────────────────────────────────────────────────────────
# Check 3 — manager-meta (extension-node-map drift)
# ───────────────────────────────────────────────────────────────────────


def check_manager_meta(actual_ids: set[str], verbose: bool = False) -> bool:
    """
    Compare ComfyUI-Manager's upstream extension-node-map entry for our repo
    against the AST-extracted node IDs. Reports phantom IDs (in upstream but
    not in our code) and missing IDs (in our code but not in upstream).
    """
    print(_cyan("manager-meta — extension-node-map.json drift"))
    if not MANAGER_EXT_MAP_URL.startswith("https://"):
        # Hardcoded URL above is https; this guard ensures any future edit
        # can't accidentally introduce a non-https scheme.
        print(_yellow("  SKIP — manager-meta URL is not https"))
        return True
    try:
        # Custom opener with a redirect handler that re-validates each hop
        # for an https scheme — closes the gap that urlopen by default would
        # follow a 30x to http:// or file:// silently.
        opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler())
        # Bandit B310: scheme is enforced by the guard above + the redirect
        # handler below; the URL is hardcoded to a github.com raw path.
        with opener.open(MANAGER_EXT_MAP_URL, timeout=20) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        print(_yellow(f"  SKIP — could not fetch upstream map: {exc}"))
        return True

    entry = data.get(REPO_URL)
    if entry is None:
        print(_yellow(f"  SKIP — no entry for {REPO_URL} in upstream map yet"))
        return True

    # Defensive shape-check: upstream entry should be `[list_of_ids, metadata]`.
    # Without this, a future schema change at upstream would silently produce
    # an empty `set()` here and report every one of our IDs as `missing`.
    if not (isinstance(entry, list) and entry and isinstance(entry[0], list)):
        print(_yellow(
            f"  SKIP — upstream entry has unexpected shape: "
            f"{type(entry).__name__}"
        ))
        return True

    upstream_ids = {x for x in entry[0] if isinstance(x, str)}

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
    Extract node `type` / `class_type` strings from a workflow JSON.
    Handles two ComfyUI export formats:
      - Canvas format: top-level dict with a `nodes` array of {type, ...}.
      - API format:    top-level dict where every value is a node spec
                       with a `class_type` key.
    Returns an empty set if the shape doesn't match either format —
    caller decides whether that's a soft or hard failure.
    """
    if not isinstance(data, dict):
        return set()

    # Canvas format
    if isinstance(data.get("nodes"), list):
        return {
            n["type"] for n in data["nodes"]
            if isinstance(n, dict) and isinstance(n.get("type"), str)
        }

    # API format — only accept if EVERY value has class_type, so a malformed
    # canvas (e.g. nodes: null) doesn't accidentally fall through and pull
    # class_type from unrelated keys.
    values = list(data.values())
    if values and all(
        isinstance(v, dict) and isinstance(v.get("class_type"), str) for v in values
    ):
        return {v["class_type"] for v in values}

    return set()


def check_workflows(actual_ids: set[str], verbose: bool = False) -> bool:
    print(_cyan("workflows — fixture node-ID stability"))
    fixtures_dir = REPO_ROOT / "tests" / "workflows"
    if not fixtures_dir.is_dir():
        print(_yellow("  SKIP — no tests/workflows/ folder yet"))
        return True

    fixtures = sorted(fixtures_dir.glob("*.json"))
    if not fixtures:
        # If the dir exists with a README but contains no *.json, that's a
        # regression in fixture coverage (someone deleted them or typoed
        # the dir name). Fail loudly rather than silently pass.
        if (fixtures_dir / "README.md").exists():
            print(_red(
                "  FAIL — fixtures dir exists with README but no *.json files. "
                "If this is intentional, delete the README too."
            ))
            return False
        print(_yellow("  SKIP — no *.json fixtures present"))
        return True

    failures: list[str] = []
    total_koolook_refs = 0

    for fx in fixtures:
        try:
            data = json.loads(fx.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{fx.name}: invalid JSON ({exc})")
            continue

        types = _extract_workflow_node_types(data)

        # Two-pronged matching:
        #   (a) heuristic match — anything that LOOKS like a Koolook ID by
        #       the name pattern. Catches drift where a fixture references
        #       a renamed/removed Koolook ID even if the fork's suffixing
        #       is opaque to the static check.
        #   (b) actual_ids match — anything in the AST-extracted set is by
        #       definition Koolook. Catches IDs that don't fit the heuristic
        #       (e.g. future Koolook nodes named without an "Easy"/"easy_"
        #       prefix and without "koolook" in the ID itself).
        # Union for the "this fixture references N Koolook IDs" report;
        # heuristic-minus-actual for the "missing ID" failure flag.
        koolook_by_heuristic = {t for t in types if _looks_like_koolook_id(t)}
        koolook_by_actual = types & actual_ids
        koolook_total = koolook_by_heuristic | koolook_by_actual
        missing = koolook_by_heuristic - actual_ids

        total_koolook_refs += len(koolook_total)

        if verbose:
            print(_dim(
                f"  {fx.name}: {len(types)} types, "
                f"{len(koolook_total)} Koolook (heuristic={len(koolook_by_heuristic)}, "
                f"actual={len(koolook_by_actual)}), "
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

    print(_green(
        f"  PASS — {len(fixtures)} fixture(s) reference "
        f"{total_koolook_refs} Koolook node ID(s), all currently registered"
    ))
    return True


# ───────────────────────────────────────────────────────────────────────
# Orchestration
# ───────────────────────────────────────────────────────────────────────


CHECKS = ("static", "dispatch", "manager-meta", "workflows")
DEFAULT_CHECKS = ("static", "dispatch", "workflows")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", choices=CHECKS, action="append",
        help=(
            "Run only the named check (repeatable). Default: run release gates "
            "(static, dispatch, workflows)."
        ),
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

    selected = set(args.check or DEFAULT_CHECKS)
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
        ok, actual_ids = check_static(verbose=args.verbose)
        results["static"] = ok
        print()

    if "dispatch" in to_run:
        results["dispatch"] = check_dispatch(verbose=args.verbose)
        print()

    if "manager-meta" in to_run:
        if not actual_ids and "static" not in to_run:
            ok, actual_ids = check_static(verbose=False)
        results["manager-meta"] = check_manager_meta(actual_ids, verbose=args.verbose)
        print()

    if "workflows" in to_run:
        if not actual_ids and "static" not in to_run:
            ok, actual_ids = check_static(verbose=False)
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
