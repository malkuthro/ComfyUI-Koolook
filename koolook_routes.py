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
    async def list_presets(_request):
        base, _is_default = _configured_dir()
        if not base.exists() or not base.is_dir():
            return web.json_response([])
        out = []
        try:
            for entry in base.iterdir():
                if not entry.is_file() or not entry.name.lower().endswith(".json"):
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

    @routes.get("/koolook/presets/file")
    async def get_preset(request):
        name = _validate_filename(request.query.get("name"))
        base, _ = _configured_dir()
        file_path = _resolve_within_base(base, name)
        if not file_path.is_file():
            raise web.HTTPNotFound(reason=f"Preset '{name}' not found.")
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not read preset: {exc}"
            ) from exc
        return web.Response(text=content, content_type="application/json")

    @routes.head("/koolook/presets/file")
    async def head_preset(request):
        """Cheap existence check used by the client's `presetExists` probe.

        Without this, aiohttp auto-synthesizes HEAD from the GET handler,
        which still reads the entire file from disk and discards the body
        — wasteful on multi-MB snapshots over Dropbox/iCloud/NFS-mounted
        libraries. The dedicated handler does just `is_file()` + status,
        no body work.
        """
        name = _validate_filename(request.query.get("name"))
        base, _ = _configured_dir()
        file_path = _resolve_within_base(base, name)
        if not file_path.is_file():
            return web.Response(status=404)
        return web.Response(status=200)

    @routes.post("/koolook/presets/file")
    async def post_preset(request):
        name = _validate_filename(request.query.get("name"))
        base, _ = _configured_dir()
        if not base.parent.exists():
            raise web.HTTPInternalServerError(
                reason=(
                    f"Preset library parent directory does not exist: "
                    f"{base.parent}. Create it (or fix the {ENV_VAR} env "
                    f"var) and retry."
                )
            )
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not create preset directory: {exc}"
            ) from exc
        body = await request.read()
        file_path = _resolve_within_base(base, name)
        try:
            file_path.write_bytes(body)
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not write preset: {exc}"
            ) from exc
        return web.json_response({"ok": True, "name": name})

    @routes.delete("/koolook/presets/file")
    async def delete_preset(request):
        name = _validate_filename(request.query.get("name"))
        base, _ = _configured_dir()
        file_path = _resolve_within_base(base, name)
        if not file_path.is_file():
            raise web.HTTPNotFound(reason=f"Preset '{name}' not found.")
        try:
            file_path.unlink()
        except OSError as exc:
            raise web.HTTPInternalServerError(
                reason=f"Could not delete preset: {exc}"
            ) from exc
        return web.json_response({"ok": True, "name": name})


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
