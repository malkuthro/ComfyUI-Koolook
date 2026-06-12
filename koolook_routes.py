"""HTTP routes for the Kforge Labs preset library.

Exposes ``/koolook/presets/*`` endpoints over ComfyUI's aiohttp server. The
preset directory is configurable via the ``KFORGELABS_PRESETS`` env var; if
unset, falls back to ``<comfyui-userdata>/koolook-presets/`` so users who
don't care about cross-machine sharing get a sensible default.

Endpoints:
    GET    /koolook/presets/info
    GET    /koolook/presets/list
    GET    /koolook/presets/file?name=<name>
    POST   /koolook/presets/file?name=<name>
    DELETE /koolook/presets/file?name=<name>

Path-traversal protection: filenames are restricted to a whitelist
``^[A-Za-z0-9 _.()\\-]+\\.json$``. Anything else returns 400. We enforce at
the server boundary so the client cannot smuggle ``../`` or absolute paths
even if the JS layer is compromised.

The configured directory is auto-created on first save (parent must already
exist — for facility shares mounted via NFS/SMB the mount point itself
must be present before saves succeed).
"""
from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path

from aiohttp import web

try:
    from .koolook_setups import PublishedSetupRegistry, default_registry
    from .koolook_setup_runner import (
        AiohttpComfyClient,
        InMemorySetupRunStore,
        PublishedSetupRunner,
        SetupRunError,
    )
except ImportError:  # pragma: no cover - standalone test/import context
    from koolook_setups import PublishedSetupRegistry, default_registry
    from koolook_setup_runner import (
        AiohttpComfyClient,
        InMemorySetupRunStore,
        PublishedSetupRunner,
        SetupRunError,
    )

ENV_VAR = "KFORGELABS_PRESETS"
DEFAULT_SUBDIR = "koolook-presets"
SETTINGS_FILENAME = "koolook-settings.json"
SETTINGS_KEY_LIBRARY_PATH = "libraryPath"

# Single-segment filename, ending in `.json`. No path separators, no `..`,
# no leading dot. Allows letters, digits, spaces, and a small set of safe
# punctuation (underscore, period, parentheses, hyphen).
_FILENAME_RE = re.compile(r"^[A-Za-z0-9 _.()\-]+\.json$")

# Same charset as filenames but without the `.json` extension — for the
# optional `dir` query param that scopes preset operations into a per-preset
# auto-save subfolder (`<preset>_autosave/`). Single segment only; no path
# separators (otherwise the symlink-escape check would have to scan multiple
# levels). The library structure is deliberately one level deep.
_DIRNAME_RE = re.compile(r"^[A-Za-z0-9 _.()\-]+$")

# Filenames the list endpoint should hide from the user-facing Load list.
# Currently used to suppress legacy flat-file autosaves left over from the
# previous autosave layout, in case a user has already accumulated some
# before upgrading to the per-preset-subfolder layout. Cheap defensive
# filter; can be removed in a future major if we choose to migrate.
_HIDDEN_LIST_PREFIXES = ("_autosave_",)
_WEB_DIR = Path(__file__).resolve().parent / "web"


def _latest_autosave_mtime(autosave_dir: Path, named_mtime: float) -> float | None:
    """Returns the newest recovery-file mtime inside ``autosave_dir`` if
    strictly newer than ``named_mtime``, else ``None``.

    Scans ``periodic.json`` and every ``pre_load_*.json``. Powers the Load
    dialog's "newer auto-save available" affordance — the dialog needs to
    surface the freshest recovery file regardless of which mechanism wrote
    it (the periodic timer vs. an explicit pre-load capture).
    """
    newest: float | None = None
    try:
        st = (autosave_dir / "periodic.json").stat()
        newest = st.st_mtime
    except OSError:
        pass
    try:
        for entry in autosave_dir.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.startswith("pre_load_") or not entry.name.endswith(".json"):
                continue
            try:
                pst = entry.stat()
            except OSError:
                continue
            if newest is None or pst.st_mtime > newest:
                newest = pst.st_mtime
    except OSError:
        pass
    if newest is not None and newest > named_mtime:
        return newest
    return None


def _resolve_default_dir() -> Path:
    """Resolve the default preset directory under ComfyUI's user folder.

    Uses ``folder_paths.get_user_directory()`` (provided by ComfyUI core)
    so we don't hardcode the userdata layout — a custom-installed Comfy
    with a non-standard user dir still gets a sensible default.
    """
    try:
        import folder_paths

        user_dir = folder_paths.get_user_directory()
        return Path(user_dir) / DEFAULT_SUBDIR
    except Exception:  # pragma: no cover - fallback for unusual setups
        return Path.cwd() / "user" / "default" / DEFAULT_SUBDIR


def _settings_file_path() -> Path:
    """Returns the path to ``koolook-settings.json`` under ComfyUI's user dir.

    The settings file is per-install (so each ComfyUI on a workstation has
    its own preferred library path). Stored as JSON with shape:
        { "libraryPath": "<absolute-path-or-empty>" }
    """
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / SETTINGS_FILENAME
    except Exception:  # pragma: no cover
        return Path.cwd() / "user" / "default" / SETTINGS_FILENAME


def _read_settings() -> dict:
    """Read the settings JSON. Returns ``{}`` if missing or unreadable.

    Soft-failure on read: the in-UI Settings panel can still rewrite a
    fresh file even if the existing one is corrupt, and the rest of the
    plugin continues to function (just falls through to env var / default).
    """
    path = _settings_file_path()
    try:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_settings(data: dict) -> None:
    """Persist the settings JSON atomically.

    Writes to ``<settings>.tmp`` then ``os.replace`` to the final name —
    so an interrupted process or concurrent write can't leave a truncated
    file that ``_read_settings`` would silently revert to ``{}`` on next
    load (silently dropping the user's saved ``libraryPath``).

    Raises ``web.HTTPInternalServerError`` on failure.
    """
    path = _settings_file_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    except OSError as exc:
        # Best-effort cleanup of the tmp file so it doesn't accumulate.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise web.HTTPInternalServerError(
            reason=f"Could not write settings file: {exc}"
        ) from exc


def _configured_dir() -> tuple[Path, str]:
    """Return ``(dir_path, source)`` where source is one of
    ``"settings"``, ``"env"``, ``"default"``.

    Resolution order (highest priority first):
      1. ``libraryPath`` field in the settings file (set via Settings panel)
      2. ``KFORGELABS_PRESETS`` env var (deployment / facility config)
      3. Built-in default ``<comfyui-userdata>/koolook-presets/``

    Reads each source on every call rather than caching at module-load,
    so a Settings panel save or a relauncher with a new env value picks
    up immediately.
    """
    settings = _read_settings()
    settings_path = settings.get(SETTINGS_KEY_LIBRARY_PATH, "")
    if isinstance(settings_path, str) and settings_path.strip():
        return Path(settings_path.strip()).expanduser(), "settings"
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser(), "env"
    return _resolve_default_dir(), "default"


def _browse_roots() -> list[Path]:
    """Return filesystem roots the in-app browser can start from."""
    if os.name == "nt":
        roots = []
        for letter in string.ascii_uppercase:
            root = Path(f"{letter}:\\")
            if root.exists():
                roots.append(root)
        return roots
    return [Path("/")]


def _browse_start_path(raw_path: str) -> Path:
    """Resolve the requested browse path or fall back to the library dir."""
    if raw_path:
        return Path(raw_path).expanduser()

    base, _ = _configured_dir()
    if base.exists():
        return base
    if base.parent.exists():
        return base.parent
    try:
        return Path.home()
    except RuntimeError:  # pragma: no cover - unusual host config
        return Path.cwd()


def _list_child_dirs(path: Path) -> list[dict]:
    out = []
    for entry in path.iterdir():
        try:
            if not entry.is_dir():
                continue
            if entry.name == "_unsaved_autosave" or entry.name.endswith("_autosave"):
                continue
            out.append({"name": entry.name, "path": str(entry)})
        except OSError:
            continue
    out.sort(key=lambda r: r["name"].lower())
    return out


def _list_child_files(path: Path) -> list[dict]:
    """List child *.json files in ``path``.

    Powers the folder picker's "yes, this is the folder I expected"
    affordance (mockup section 6) — files are rendered greyed so the user
    can visually confirm the listing without selecting one. Restricted to
    JSON because the picker is exclusively used to choose a snapshot
    library folder; surfacing non-snapshot files would be misleading
    (they cannot become preset library content).
    """
    out = []
    for entry in path.iterdir():
        try:
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(".json"):
                continue
        except OSError:
            continue
        out.append({"name": entry.name})
    out.sort(key=lambda r: r["name"].lower())
    return out


def _validate_new_dir_name(name: object) -> str:
    """Validate a user-created folder name for the browse picker."""
    if not isinstance(name, str):
        raise web.HTTPBadRequest(reason="Folder name must be a string.")
    cleaned = name.strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
    ):
        raise web.HTTPBadRequest(reason="Folder name must be a single folder segment.")
    return cleaned


def _validate_filename(name: object) -> str:
    """Whitelist the filename or raise 400.

    Returning the validated string is convenient for direct
    ``base / _validate_filename(...)`` use at call sites.
    """
    if not isinstance(name, str) or not _FILENAME_RE.match(name):
        raise web.HTTPBadRequest(
            reason=(
                "Invalid filename. Allowed characters: letters, digits, "
                "space, underscore, period, parentheses, hyphen. Must end "
                "in .json."
            )
        )
    return name


def _validate_dirname(name: str) -> str:
    """Whitelist a single-segment subfolder name or raise 400.

    Same charset as filenames but without the `.json` extension. Used by
    the autosave subfolder routing — a request with ``?dir=<x>`` lands its
    file inside ``<library>/<x>/`` instead of the library root.
    """
    if not isinstance(name, str) or not _DIRNAME_RE.match(name):
        raise web.HTTPBadRequest(
            reason=(
                "Invalid dir. Allowed characters: letters, digits, space, "
                "underscore, period, parentheses, hyphen. No path "
                "separators, no extension."
            )
        )
    return name


def _resolve_within_base(base: Path, name: str) -> Path:
    """Compute ``base / name`` and verify the resolved path stays under
    ``base`` after symlinks are followed. Defense-in-depth against a
    symlink planted inside the configured library dir (plausible on a
    facility shared mount with weak permissions) redirecting reads/writes
    off-share.

    Raises ``web.HTTPBadRequest`` on any escape (broken or hostile symlink).
    Returns the joined path on success — note this is the un-resolved path
    so the caller can still see the intended filename in error messages,
    but the ``is_relative_to`` check has already been applied to the
    resolved form.
    """
    file_path = base / name
    try:
        resolved_file = file_path.resolve(strict=False)
        resolved_base = base.resolve(strict=False)
    except OSError as exc:
        raise web.HTTPInternalServerError(
            reason=f"Could not resolve preset path: {exc}"
        ) from exc
    try:
        resolved_file.relative_to(resolved_base)
    except ValueError as exc:
        raise web.HTTPBadRequest(
            reason=(
                "Preset path escapes the library directory (likely a symlink "
                "planted in the library). Refusing to proceed."
            )
        ) from exc
    return file_path


def _resolve_target(lib_base: Path, subdir: str, name: str) -> tuple[Path, Path]:
    """Resolve ``lib_base[/subdir]/name`` for a preset operation, with the
    symlink-escape check grounded at ``lib_base`` (NOT at the subdir) so a
    hostile symlink inside an auto-save subfolder can't redirect writes
    outside the library.

    Returns ``(target_dir, file_path)`` — the caller may need ``target_dir``
    to mkdir before writing. ``file_path`` is the un-resolved join, same as
    ``_resolve_within_base``.

    Raises ``web.HTTPBadRequest`` on escape, ``web.HTTPInternalServerError``
    on resolution failure.
    """
    if subdir:
        target_dir = lib_base / subdir
    else:
        target_dir = lib_base
    file_path = target_dir / name
    try:
        resolved_file = file_path.resolve(strict=False)
        resolved_lib = lib_base.resolve(strict=False)
    except OSError as exc:
        raise web.HTTPInternalServerError(
            reason=f"Could not resolve preset path: {exc}"
        ) from exc
    try:
        resolved_file.relative_to(resolved_lib)
    except ValueError as exc:
        raise web.HTTPBadRequest(
            reason=(
                "Preset path escapes the library directory (likely a symlink "
                "planted in the library or subfolder). Refusing to proceed."
            )
        ) from exc
    return target_dir, file_path


def _open_dir_in_file_manager(target_dir: Path) -> None:
    """Open ``target_dir`` in the OS file manager. Shared by the snapshot
    and published-setup reveal routes. Raises ``web.HTTPNotFound`` when the
    directory is absent and ``web.HTTPInternalServerError`` if the launcher
    cannot start. Subprocess args are passed list-form (no shell), so a
    controlled path can't trigger shell-metacharacter interpretation.
    """
    if not target_dir.exists() or not target_dir.is_dir():
        raise web.HTTPNotFound(reason=f"Path does not exist on disk: {target_dir}")
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target_dir)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer.exe", str(target_dir)])
        else:
            subprocess.Popen(["xdg-open", str(target_dir)])
    except OSError as exc:
        raise web.HTTPInternalServerError(
            reason=f"Could not open path in file manager: {exc}"
        ) from exc


def register_routes(routes, setup_registry_factory=None, setup_runner_factory=None) -> None:
    """Attach the preset endpoints to the given aiohttp ``RouteTableDef``.

    Called once from ``__init__.py`` at custom-node load time. Splitting
    registration from the route handlers keeps this file unit-testable
    without an aiohttp app fixture.
    """
    if setup_registry_factory is None:
        setup_registry_factory = default_registry
    run_store = InMemorySetupRunStore()

    def _default_setup_runner(request, registry: PublishedSetupRegistry) -> PublishedSetupRunner:
        base_url = f"{request.scheme}://{request.host}"
        return PublishedSetupRunner(registry, AiohttpComfyClient(base_url), run_store)

    def _setup_runner(request, registry: PublishedSetupRegistry):
        if setup_runner_factory is not None:
            return setup_runner_factory()
        return _default_setup_runner(request, registry)

    def _log_setup_diagnostics(registry: PublishedSetupRegistry) -> None:
        for diagnostic in registry.diagnostics:
            print(f"[Koolook] published setup skipped: {diagnostic}")

    def _web_asset_response(filename: str, content_type: str):
        path = _WEB_DIR / filename
        try:
            return web.Response(
                text=path.read_text(encoding="utf-8"),
                content_type=content_type,
            )
        except OSError as exc:
            raise web.HTTPNotFound(reason=f"Koolook web asset '{filename}' not found.") from exc

    @routes.get("/koolook/setup_runner_simulator.html")
    async def setup_runner_simulator_html(_request):
        return _web_asset_response("setup_runner_simulator.html", "text/html")

    @routes.get("/koolook/setup_runner_simulator.js")
    async def setup_runner_simulator_js(_request):
        return _web_asset_response("setup_runner_simulator.js", "application/javascript")

    @routes.get("/koolook/api/setups")
    async def list_published_setups(_request):
        registry: PublishedSetupRegistry = setup_registry_factory()
        rows = registry.listSetups()
        _log_setup_diagnostics(registry)
        return web.json_response(rows)

    @routes.get("/koolook/api/setups/{setup_id}")
    async def get_published_setup(request):
        registry: PublishedSetupRegistry = setup_registry_factory()
        setup_id = request.match_info["setup_id"]
        setup = registry.getSetup(setup_id)
        _log_setup_diagnostics(registry)
        if setup is None:
            raise web.HTTPNotFound(reason=f"Published setup '{setup_id}' not found.")
        return web.json_response(setup)

    @routes.post("/koolook/api/setups")
    async def publish_setup(request):
        try:
            payload = await request.json()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "errors": [f"Body must be JSON: {exc}"]},
                status=400,
            )
        if not isinstance(payload, dict):
            return web.json_response(
                {"ok": False, "errors": ["Body must be a JSON object."]},
                status=400,
            )

        registry: PublishedSetupRegistry = setup_registry_factory()
        result = registry.publishSetup(
            visualGraph=payload.get("visualGraph"),
            metadata=payload.get("metadata"),
            inputContract=payload.get("inputContract"),
            outputContract=payload.get("outputContract"),
            source=payload.get("source"),
            apiPrompt=payload.get("apiPrompt"),
        )
        if not result.valid:
            return web.json_response(
                {"ok": False, "errors": result.diagnostics},
                status=400,
            )
        response = {"ok": True, "setup": result.setup}
        storage_path = registry.storage_path
        if storage_path is not None:
            response["storagePath"] = str(storage_path)
        return web.json_response(response)

    @routes.post("/koolook/api/setups/{setup_id}/run")
    async def run_published_setup(request):
        try:
            payload = await request.json()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "errors": [f"Body must be JSON: {exc}"]},
                status=400,
            )
        if not isinstance(payload, dict):
            return web.json_response(
                {"ok": False, "errors": ["Body must be a JSON object."]},
                status=400,
            )
        registry: PublishedSetupRegistry = setup_registry_factory()
        runner = _setup_runner(request, registry)
        try:
            result = await runner.runSetup(
                request.match_info["setup_id"],
                payload.get("inputs", {}),
            )
        except SetupRunError as exc:
            return web.json_response(
                {"ok": False, "errors": exc.errors},
                status=exc.status_code,
            )
        return web.json_response({"ok": True, "run": result})

    @routes.get("/koolook/api/runs/{run_id}")
    async def get_published_setup_run(request):
        registry: PublishedSetupRegistry = setup_registry_factory()
        runner = _setup_runner(request, registry)
        try:
            result = await runner.getRun(request.match_info["run_id"])
        except SetupRunError as exc:
            return web.json_response(
                {"ok": False, "errors": exc.errors},
                status=exc.status_code,
            )
        return web.json_response({"ok": True, "run": result})

    @routes.get("/koolook/presets/info")
    async def info(_request):
        base, source = _configured_dir()
        exists = base.exists() and base.is_dir()
        writable = exists and os.access(str(base), os.W_OK)
        return web.json_response(
            {
                "path": str(base),
                "source": source,           # "settings" | "env" | "default"
                "isDefault": source == "default",
                "envVar": ENV_VAR,
                "exists": exists,
                "writable": writable,
            }
        )

    @routes.get("/koolook/presets/settings")
    async def get_settings(_request):
        """Return the saved-in-UI library path (if any) plus the resolved
        path the server actually uses right now. The UI's Settings dialog
        renders the saved value as the editable field and the resolved
        value as the read-only "currently in effect" line."""
        settings = _read_settings()
        saved = settings.get(SETTINGS_KEY_LIBRARY_PATH, "")
        if not isinstance(saved, str):
            saved = ""
        resolved, source = _configured_dir()
        return web.json_response(
            {
                "savedLibraryPath": saved,
                "resolvedPath": str(resolved),
                "source": source,
                "envVar": ENV_VAR,
            }
        )

    @routes.get("/koolook/presets/browse")
    async def browse_dirs(request):
        """List child directories for the Settings dialog's path picker.

        With ``?files=1`` also returns child ``*.json`` files — used by the
        redesigned folder picker (issue #137, mockup section 6) to render a
        greyed "this is the folder I expected" affordance below the
        directories. Default response shape stays directory-only so any
        existing callers (legacy Settings dialog) continue to work
        unchanged.

        The selected path is still saved through the existing settings
        endpoint, so the library-path resolution and write checks stay
        centralized.
        """
        path = _browse_start_path(request.query.get("path", "").strip())
        include_files = request.query.get("files", "").strip() in {"1", "true", "yes"}
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise web.HTTPBadRequest(reason=f"Could not resolve path: {exc}") from exc
        if not resolved.exists() or not resolved.is_dir():
            raise web.HTTPBadRequest(reason=f"Directory does not exist: {resolved}")

        parent = resolved.parent if resolved.parent != resolved else None
        roots = [{"name": str(root), "path": str(root)} for root in _browse_roots()]
        try:
            dirs = _list_child_dirs(resolved)
            files = _list_child_files(resolved) if include_files else []
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not list directories: {exc}"
            ) from exc
        body = {
            "path": str(resolved),
            "parentPath": str(parent) if parent else "",
            "roots": roots,
            "dirs": dirs,
        }
        if include_files:
            body["files"] = files
        return web.json_response(body)

    @routes.post("/koolook/presets/browse/new-folder")
    async def create_browse_dir(request):
        """Create one child folder under the current browse location.

        Threat model: this picker is intentionally filesystem-wide so users
        can choose a snapshot library outside the current Koolook library.
        The route's boundary is the ComfyUI process OS user, not the preset
        library root. We still reject multi-segment names and symlink escapes
        relative to the selected parent so a request cannot turn "create this
        child folder here" into "create something elsewhere."
        """
        try:
            payload = await request.json()
        except Exception as exc:
            raise web.HTTPBadRequest(reason=f"Body must be JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="Body must be a JSON object.")

        parent = _browse_start_path(str(payload.get("parentPath", "")).strip())
        name = _validate_new_dir_name(payload.get("name", ""))
        try:
            resolved_parent = parent.resolve(strict=False)
        except OSError as exc:
            raise web.HTTPBadRequest(reason=f"Could not resolve parent: {exc}") from exc
        if not resolved_parent.exists() or not resolved_parent.is_dir():
            raise web.HTTPBadRequest(reason=f"Parent folder does not exist: {resolved_parent}")

        child = resolved_parent / name
        try:
            resolved_child = child.resolve(strict=False)
        except OSError as exc:
            raise web.HTTPBadRequest(reason=f"Could not resolve new folder: {exc}") from exc
        try:
            resolved_child.relative_to(resolved_parent)
        except ValueError as exc:
            raise web.HTTPBadRequest(reason="New folder escapes the selected parent.") from exc
        if resolved_child.exists():
            raise web.HTTPBadRequest(reason=f"Folder already exists: {resolved_child}")

        try:
            child.mkdir()
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not create folder: {exc}"
            ) from exc
        return web.json_response({"path": str(child)})

    @routes.post("/koolook/presets/settings")
    async def post_settings(request):
        """Persist a saved library path. Body is JSON ``{libraryPath: <str>}``.

        Empty string or missing field clears the override (the server then
        falls back to env-var or the built-in default).
        """
        try:
            payload = await request.json()
        except Exception as exc:
            raise web.HTTPBadRequest(
                reason=f"Settings body must be JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="Settings body must be a JSON object.")
        new_path = payload.get(SETTINGS_KEY_LIBRARY_PATH, "")
        if not isinstance(new_path, str):
            raise web.HTTPBadRequest(
                reason=f"`{SETTINGS_KEY_LIBRARY_PATH}` must be a string."
            )
        new_path = new_path.strip()
        # Read-modify-write so unrelated keys (future settings) survive.
        existing = _read_settings()
        if new_path:
            existing[SETTINGS_KEY_LIBRARY_PATH] = new_path
        else:
            # Empty string = clear the override; remove the key entirely
            # so the resolution chain falls through cleanly to env / default.
            existing.pop(SETTINGS_KEY_LIBRARY_PATH, None)
        _write_settings(existing)
        # Echo the new resolved state so the UI can update without a refetch.
        resolved, source = _configured_dir()
        return web.json_response(
            {
                "savedLibraryPath": new_path,
                "resolvedPath": str(resolved),
                "source": source,
            }
        )

    @routes.get("/koolook/presets/list")
    async def list_presets(request):
        """List preset files. With ``?dir=<x>`` lists files inside that
        subfolder (used by autosave management); without dir, lists root
        and HIDES anything matching ``_HIDDEN_LIST_PREFIXES`` (legacy flat
        autosaves) plus any subdirectories (the per-preset autosave folders
        — those should never appear in the user-facing Load list).
        """
        base, _is_default = _configured_dir()
        subdir_q = request.query.get("dir", "").strip()
        if subdir_q:
            _validate_dirname(subdir_q)
            target_dir, _ = _resolve_target(base, subdir_q, "_listing.json")
            target_dir = target_dir  # named for clarity
        else:
            target_dir = base
        if not target_dir.exists() or not target_dir.is_dir():
            return web.json_response([])
        out = []
        try:
            for entry in target_dir.iterdir():
                if not entry.is_file() or not entry.name.lower().endswith(".json"):
                    continue
                # Only filter legacy autosave flat files at the LIBRARY ROOT.
                # Inside autosave subfolders we want to see everything (so
                # the rotation pruner can list + delete its own files).
                if not subdir_q and any(
                    entry.name.startswith(p) for p in _HIDDEN_LIST_PREFIXES
                ):
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                row = {
                    "name": entry.name,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
                # Library-root rows only: scan the matching
                # `<base>_autosave/` subfolder for the newest recovery file
                # (``periodic.json`` and every ``pre_load_*.json``) and
                # surface its mtime when strictly newer than the named
                # file. Powers the Load dialog's inline "Newer auto-save
                # available — restore?" affordance — the autosave UX needs
                # the system to proactively offer the recovery copy when
                # the user's last manual save is stale, rather than
                # burying it inside the Recovery disclosure. Issue #137:
                # before this we only considered ``periodic.json``, which
                # missed cases where the user's most recent Load wrote a
                # fresher ``pre_load_*.json`` snapshot. A handful of stats
                # per row, no reads — still cheap on large libraries.
                if not subdir_q:
                    base_name = entry.name[: -len(".json")]
                    autosave_dir = target_dir / f"{base_name}_autosave"
                    newest = _latest_autosave_mtime(autosave_dir, stat.st_mtime)
                    if newest is not None:
                        row["latestAutosaveMtime"] = newest
                out.append(row)
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not list preset library: {exc}"
            ) from exc
        # Sort alphabetical (case-insensitive) at the server so every client
        # sees the same order regardless of filesystem iteration order.
        out.sort(key=lambda r: r["name"].lower())
        return web.json_response(out)

    @routes.get("/koolook/presets/autosaves/list")
    async def list_autosaves(_request):
        """Walk every ``*_autosave/`` subdir (and ``_unsaved_autosave/``)
        under the library and return a flat list of recovery snapshot
        files. Powers the Load dialog's recovery section so a user can
        restore from a pre-load or periodic auto-save without leaving
        the UI.

        Why a dedicated endpoint instead of recursing the existing
        ``list``? The user-facing list MUST hide subdirs (otherwise the
        Load list gets cluttered with autosave files that aren't
        directly user-restorable). This endpoint is the explicit "show
        me everything in autosave folders" API — separated cleanly so
        the boundary is grep-able and adding more subdir-scoped views
        later (timeline / cross-preset comparison / etc.) doesn't have
        to push more flags into the main list endpoint.

        Sort: subdir alphabetical, then mtime descending within each
        subdir (newest recovery point first — matches what users
        expect when scanning for "the file just before I broke things").
        """
        base, _ = _configured_dir()
        if not base.exists() or not base.is_dir():
            return web.json_response([])
        out = []
        try:
            for subdir_entry in base.iterdir():
                if not subdir_entry.is_dir():
                    continue
                subdir_name = subdir_entry.name
                # Match the autosave naming convention exactly so a user-
                # created subdirectory called "my random folder" isn't
                # treated as a recovery source.
                is_unsaved = subdir_name == "_unsaved_autosave"
                is_named = subdir_name.endswith("_autosave") and len(subdir_name) > len("_autosave")
                if not (is_unsaved or is_named):
                    continue
                for f in subdir_entry.iterdir():
                    if not f.is_file() or not f.name.lower().endswith(".json"):
                        continue
                    try:
                        stat = f.stat()
                    except OSError:
                        continue
                    out.append(
                        {
                            "dir": subdir_name,
                            "name": f.name,
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }
                    )
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not list autosave folders: {exc}"
            ) from exc
        out.sort(key=lambda r: (r["dir"].lower(), -r["mtime"]))
        return web.json_response(out)

    @routes.get("/koolook/presets/file")
    async def get_preset(request):
        """Serve preset JSON for ``?name=<name>[&dir=<subdir>]``; HEAD
        short-circuits before the file read.

        aiohttp's ``add_get`` auto-registers a HEAD route by default. We
        can't opt out via ``allow_head=False`` because ComfyUI's
        mirror-to-/api code in ``server.py`` blindly forwards
        ``RouteDef.kwargs`` into ``RouteTableDef.route(method, path)
        (handler, **kwargs)`` whose inner closure rejects unknown
        kwargs, crashing startup. So this single handler owns both
        methods, branching on ``request.method`` so a ``presetExists``
        HEAD probe doesn't have to read multi-MB JSON over
        Dropbox/iCloud/NFS-mounted libraries just to answer 200/404.
        """
        name = _validate_filename(request.query.get("name"))
        subdir = request.query.get("dir", "").strip()
        if subdir:
            _validate_dirname(subdir)
        base, _ = _configured_dir()
        _, file_path = _resolve_target(base, subdir, name)
        if not file_path.is_file():
            raise web.HTTPNotFound(reason=f"Preset '{name}' not found.")
        if request.method == "HEAD":
            return web.Response(status=200)
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not read preset: {exc}"
            ) from exc
        return web.Response(text=content, content_type="application/json")

    @routes.post("/koolook/presets/file")
    async def post_preset(request):
        name = _validate_filename(request.query.get("name"))
        subdir = request.query.get("dir", "").strip()
        if subdir:
            _validate_dirname(subdir)
        base, _ = _configured_dir()
        if not base.parent.exists():
            raise web.HTTPInternalServerError(
                reason=(
                    f"Preset library parent directory does not exist: "
                    f"{base.parent}. Create it (or fix the {ENV_VAR} env "
                    f"var) and retry."
                )
            )
        target_dir, file_path = _resolve_target(base, subdir, name)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not create preset directory: {exc}"
            ) from exc
        body = await request.read()
        # Atomic write — `tempfile.mkstemp` opens a uniquely-named temp
        # file inside the already-validated `target_dir` using
        # `O_CREAT | O_EXCL` semantics, then `os.replace` swaps it onto
        # the final name. A crash or dropped connection mid-write can
        # no longer truncate the existing good file: readers see either
        # the old or the new content, never a half-written byte stream.
        #
        # Why mkstemp instead of a fixed `<file>.json.tmp` name? An
        # attacker (or accidental user) with write access to the library
        # could plant `<file>.json.tmp` as a symlink pointing outside
        # the library. A naive `Path.write_bytes` would FOLLOW that
        # symlink — past the path-traversal guard — and write the
        # request body to wherever the symlink pointed, before
        # `os.replace` swapped the symlink off. `mkstemp` defeats that
        # in two ways: (1) `O_EXCL` aborts if anything (including a
        # symlink) already exists at the chosen name, and (2) the name
        # itself is random so the attacker can't pre-plant anything
        # there. The `.tmp` suffix keeps the transient file invisible
        # to `/koolook/presets/list` and `/koolook/presets/autosaves/list`
        # (both filter to `.json`).
        try:
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=f"{file_path.stem}.",
                suffix=".tmp",
                dir=str(target_dir),
            )
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not create preset temp file: {exc}"
            ) from exc
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(body)
            os.replace(tmp_name, str(file_path))
        except OSError as exc:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
            raise web.HTTPInternalServerError(
                reason=f"Could not write preset: {exc}"
            ) from exc
        return web.json_response({"ok": True, "name": name, "dir": subdir})

    @routes.delete("/koolook/presets/file")
    async def delete_preset(request):
        name = _validate_filename(request.query.get("name"))
        subdir = request.query.get("dir", "").strip()
        if subdir:
            _validate_dirname(subdir)
        base, _ = _configured_dir()
        _, file_path = _resolve_target(base, subdir, name)
        if not file_path.is_file():
            raise web.HTTPNotFound(reason=f"Preset '{name}' not found.")
        try:
            file_path.unlink()
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not delete preset: {exc}"
            ) from exc
        return web.json_response({"ok": True, "name": name, "dir": subdir})

    @routes.post("/koolook/presets/reveal")
    async def reveal_preset_folder(request):
        """Open the preset library (or an autosave subfolder) in the OS
        file manager. Optional ``?dir=<subdir>`` deep-links to a
        per-preset autosave folder so the Recovery section can drop the
        user inside the right `<preset>_autosave/` directory.

        Path-traversal is grounded at the configured library base via
        ``_resolve_target``; subdir names go through ``_validate_dirname``.
        Subprocess args are passed list-form (no shell), so a controlled
        path can't trigger shell-metacharacter interpretation even if a
        future check loosens the dirname charset.
        """
        base, _ = _configured_dir()
        subdir_q = request.query.get("dir", "").strip()
        if subdir_q:
            _validate_dirname(subdir_q)
            target_dir, _ = _resolve_target(base, subdir_q, "_listing.json")
        else:
            target_dir = base
        _open_dir_in_file_manager(target_dir)
        return web.json_response({"ok": True, "path": str(target_dir)})

    @routes.post("/koolook/api/setups/reveal")
    async def reveal_published_setup_folder(_request):
        """Open the published-setups directory (where ``Publish setup``
        writes ``setups.json``) in the OS file manager. Distinct from the
        snapshot-library reveal above: the publish success card's Open
        folder action must land in the registry folder, not the snapshot
        library.
        """
        registry: PublishedSetupRegistry = setup_registry_factory()
        storage_path = registry.storage_path
        if storage_path is None:
            raise web.HTTPNotFound(
                reason="Published setup storage path is unavailable."
            )
        target_dir = storage_path.parent
        _open_dir_in_file_manager(target_dir)
        return web.json_response({"ok": True, "path": str(target_dir)})


def install() -> bool:
    """Hook the routes into ComfyUI's running PromptServer.

    Returns ``True`` on success, ``False`` if PromptServer wasn't reachable
    (very early import, headless test, etc.). The caller logs the failure
    so the rest of Koolook still loads — the snapshot feature just won't
    work in that session.
    """
    try:
        from server import PromptServer  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        register_routes(PromptServer.instance.routes)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[Koolook] failed to register preset routes: {exc}")
        return False
    return True
