"""
Stub-VAE dispatch test for the v2_3_3 wrapper, extracted from
`preflight_release.py` to keep the main script under the project's
~400-line file-size guideline.

This module:
  - Builds a `FakeTensor` (`FT`) class that mimics just enough of
    `torch.Tensor` to drive the wrapper's dispatch logic (no real math).
  - Builds a `torch` stub module + `torch.nn` / `torch.nn.functional`
    submodule stubs.
  - Loads the v2_3_3 wrapper code (`color_helpers.py` + `nodes_vae.py`)
    via `importlib` against the stub torch.
  - Runs five test cases through `Easy_hdr_VAE_encode` / `_decode`,
    asserting that the per-case dispatch path appears in `debug_info`.
  - Cleans up `sys.modules` in a `finally` block — including the
    `_pf_pkg*` entries — so a partial-load failure can't leak the stub
    torch into anything else running in the same process.

Public surface: `run(verbose: bool, repo_root: Path) -> tuple[bool, list[str]]`.
The caller (in `preflight_release.py`) is responsible for the `[2/4] …`
header and the PASS/FAIL line; this module just executes and reports.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


# ───────────────────────────────────────────────────────────────────────
# FakeTensor stub
# ───────────────────────────────────────────────────────────────────────


class FakeTensor:
    """Tensor-shaped duck type that records its shape and supports the
    handful of operations the v2_3_3 wrapper does at dispatch time."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.dtype = "float32"
        self.device = "cpu"

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def T(self):
        s = list(self.shape)
        if len(s) >= 2:
            s[-1], s[-2] = s[-2], s[-1]
        return FakeTensor(s)

    def clone(self):
        return FakeTensor(self.shape)

    def float(self):
        return self

    def __getitem__(self, key):
        # Slicing patterns the wrapper actually uses:
        #   pixels[:, fi, ...]      → drop dim 1, keep rest
        #   latent[:, :, fi, ...]   → drop dim 2, keep rest
        #   img[..., :3]            → keep leading dims, last dim = 3
        if isinstance(key, tuple) and len(key) == 3 and key[2] is Ellipsis:
            return FakeTensor((self.shape[0],) + self.shape[2:])
        if isinstance(key, tuple) and len(key) == 4 and key[3] is Ellipsis:
            return FakeTensor((self.shape[0], self.shape[1]) + self.shape[3:])
        if isinstance(key, tuple) and key[0] is Ellipsis:
            return FakeTensor(self.shape[:-1] + (3,))
        return self

    def reshape(self, *a):
        return FakeTensor(a)

    def to(self, **k):
        return self

    def __mul__(self, o): return self
    def __rmul__(self, o): return self
    def __pow__(self, o): return self
    def __sub__(self, o): return self
    def clamp(self, *a, **k): return self

    def __matmul__(self, o):
        return FakeTensor(self.shape[:-1] + (o.shape[-1],))


# ───────────────────────────────────────────────────────────────────────
# torch stub
# ───────────────────────────────────────────────────────────────────────


def _build_torch_stub() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Returns the (`torch`, `torch.nn`, `torch.nn.functional`) stub triple."""
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = FakeTensor
    torch_stub.stack = lambda ts, dim=0: (
        lambda s: (s.insert(dim, len(ts)), FakeTensor(s))[1]
    )(list(ts[0].shape))
    torch_stub.cat = lambda ts, dim=0: (
        lambda s: (s.__setitem__(dim, sum(x.shape[dim] for x in ts)), FakeTensor(s))[1]
    )(list(ts[0].shape))
    torch_stub.where = lambda c, a, b: a
    torch_stub.sign = lambda x: x
    torch_stub.abs = lambda x: x
    torch_stub.pow = lambda x, p: x
    torch_stub.tanh = lambda x: x
    torch_stub.clamp = lambda x, *a, **k: x
    torch_stub.tensor = lambda *a, **k: FakeTensor((3, 3))
    torch_stub.float32 = "float32"
    torch_stub.zeros = lambda shape, dtype=None, device=None: FakeTensor(shape)

    nn_F = types.ModuleType("torch.nn.functional")
    nn_F.pad = lambda x, *a, **k: x
    nn_F.avg_pool2d = lambda x, *a, **k: x

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.functional = nn_F
    torch_stub.nn = nn_mod

    return torch_stub, nn_mod, nn_F


# ───────────────────────────────────────────────────────────────────────
# Stub VAE classes
# ───────────────────────────────────────────────────────────────────────


class Fake2DVAE:
    """A vanilla 2D image VAE — no `latent_dim` attribute, so the wrapper
    treats it as the standard SD/SDXL/Flux case."""

    def encode(self, pixels):
        return FakeTensor(
            (1, 16, pixels.shape[1] // 8, pixels.shape[2] // 8)
        )

    def decode(self, lat):
        return FakeTensor(
            (lat.shape[0], lat.shape[2] * 8, lat.shape[3] * 8, 3)
        )


class Fake3DVAE:
    """A 3D-aware video VAE — exposes `latent_dim = 3` so the wrapper
    detects it and short-circuits the per-frame iteration on encode."""

    latent_dim = 3

    def encode(self, pixels):
        if pixels.ndim == 5:
            return FakeTensor((
                1, 16,
                pixels.shape[1] // 4,
                pixels.shape[2] // 8,
                pixels.shape[3] // 8,
            ))
        return FakeTensor((1, 16, pixels.shape[1] // 8, pixels.shape[2] // 8))

    def decode(self, lat):
        if lat.ndim == 5:
            return FakeTensor((
                1, lat.shape[2] * 4,
                lat.shape[3] * 8, lat.shape[4] * 8, 3,
            ))
        return FakeTensor((lat.shape[0], lat.shape[2] * 8, lat.shape[3] * 8, 3))


# ───────────────────────────────────────────────────────────────────────
# Public entry point
# ───────────────────────────────────────────────────────────────────────


def run(verbose: bool, repo_root: Path) -> tuple[bool, list[str]]:
    """Run the five dispatch test cases and report.

    Returns:
        (success, failures)
        - `success`: True if every case passed.
        - `failures`: list of "<case_name>: <error>" strings (empty on success).

    On success, a per-case "OK" line is printed when `verbose=True`.
    Header / PASS-FAIL formatting is the caller's responsibility — this
    module is the mechanical part.
    """
    torch_stub, nn_mod, nn_F = _build_torch_stub()
    pkg_path = repo_root / "forks" / "radiance_koolook" / "versions" / "v2_3_3"

    # Track every sys.modules key we add so the finally block can pop them
    # cleanly even if exec_module / constructors raise.
    injected_modules: list[str] = []

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

        return _run_cases(nv_mod, verbose)
    finally:
        for mod in injected_modules:
            sys.modules.pop(mod, None)


def _run_cases(nv_mod, verbose: bool) -> tuple[bool, list[str]]:
    """The actual five dispatch test cases. Receives the loaded
    `nodes_vae` module — which now sees our stub torch."""
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

    def case_4d_image():
        lat, dbg, _ = enc.encode(FakeTensor((2, 64, 64, 3)), Fake2DVAE(), **common_enc)
        assert lat["samples"].ndim == 4 and "path=image" in dbg

    def case_5d_2dvae():
        lat, dbg, _ = enc.encode(FakeTensor((1, 8, 64, 64, 3)), Fake2DVAE(), **common_enc)
        assert (
            lat["samples"].ndim == 5
            and lat["samples"].shape[2] == 8
            and "path=video-iter" in dbg
        )

    def case_5d_3dvae():
        lat, dbg, _ = enc.encode(FakeTensor((1, 8, 64, 64, 3)), Fake3DVAE(), **common_enc)
        assert "path=video-3d" in dbg

    def case_5d_lat_decode():
        img, dbg = dec.decode(
            {"samples": FakeTensor((1, 16, 8, 8, 8))}, Fake2DVAE(), **common_dec
        )
        assert img.ndim == 4 and img.shape[0] == 8 and "path=video-iter" in dbg

    def case_4d_lat_decode():
        img, dbg = dec.decode(
            {"samples": FakeTensor((2, 16, 8, 8))}, Fake2DVAE(), **common_dec
        )
        assert "path=image" in dbg

    cases = [
        ("4D image, 2D-VAE",       case_4d_image),
        ("5D + 2D-VAE iter",       case_5d_2dvae),
        ("5D + 3D-VAE pass",       case_5d_3dvae),
        ("5D latent decode iter",  case_5d_lat_decode),
        ("4D latent decode pass",  case_4d_lat_decode),
    ]

    failures: list[str] = []
    for name, fn in cases:
        try:
            fn()
            if verbose:
                # Caller can repaint this with color if it wants —
                # here we just emit a plain line so the module stays
                # color-helper-free.
                print(f"    OK {name}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    return (not failures), failures
