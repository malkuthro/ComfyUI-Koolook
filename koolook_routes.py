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

import os
import re
from pathlib import Path

from aiohttp import web

ENV_VAR = "KFORGELABS_PRESETS"
DEFAULT_SUBDIR = "koolook-presets"

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


def _configured_dir() -> tuple[Path, bool]:
    """Return ``(dir_path, is_default_fallback)``.

    Reads the env var on every call rather than caching it at module-load
    time, so a relauncher that exports a new value picks it up without
    needing a ComfyUI restart that bypasses Python's import cache.
    """
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser(), False
    return _resolve_default_dir(), True


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


def register_routes(routes) -> None:
    """Attach the preset endpoints to the given aiohttp ``RouteTableDef``.

    Called once from ``__init__.py`` at custom-node load time. Splitting
    registration from the route handlers keeps this file unit-testable
    without an aiohttp app fixture.
    """

    @routes.get("/koolook/presets/info")
    async def info(_request):
        base, is_default = _configured_dir()
        exists = base.exists() and base.is_dir()
        writable = exists and os.access(str(base), os.W_OK)
        return web.json_response(
            {
                "path": str(base),
                "isDefault": is_default,
                "envVar": ENV_VAR,
                "exists": exists,
                "writable": writable,
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
        file_path = base / name
        if not file_path.is_file():
            raise web.HTTPNotFound(reason=f"Preset '{name}' not found.")
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
        file_path = base / name
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
        file_path = base / name
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
