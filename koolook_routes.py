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
from pathlib import Path

from aiohttp import web

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


def register_routes(routes) -> None:
    """Attach the preset endpoints to the given aiohttp ``RouteTableDef``.

    Called once from ``__init__.py`` at custom-node load time. Splitting
    registration from the route handlers keeps this file unit-testable
    without an aiohttp app fixture.
    """

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

        This intentionally returns directory names only, not files. The
        selected path is still saved through the existing settings endpoint,
        so the library-path resolution and write checks stay centralized.
        """
        path = _browse_start_path(request.query.get("path", "").strip())
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
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not list directories: {exc}"
            ) from exc
        return web.json_response(
            {
                "path": str(resolved),
                "parentPath": str(parent) if parent else "",
                "roots": roots,
                "dirs": dirs,
            }
        )

    @routes.post("/koolook/presets/browse/new-folder")
    async def create_browse_dir(request):
        """Create one child folder under the current browse location."""
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
                out.append(
                    {
                        "name": entry.name,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
                )
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
        try:
            file_path.write_bytes(body)
        except OSError as exc:
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
