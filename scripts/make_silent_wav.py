"""Write a silent WAV file for LTX-2.3 (and other audio-required) graphs.

Usage:
    python scripts/make_silent_wav.py [outfile] [--seconds N]
                                      [--rate HZ] [--channels N]

Produces a stereo, 44.1 kHz, 16-bit silent track by default — the same
waveform format ComfyUI's AUDIO type expects, i.e. what the LTX Director
generates internally (`torch.zeros((1, 2, samples))` @ 44100). Drop the
output into ComfyUI's `input/` folder and load it with the stock
`LoadAudio` node when you need to satisfy a required AUDIO input but have
no real audio to feed.

Make it at least as long as your clip; LTX will trim/loop to fit.

Examples:
    # 10s silent.wav in the current folder (defaults)
    python scripts/make_silent_wav.py

    # 30s, written straight into a ComfyUI input dir
    python scripts/make_silent_wav.py "C:/ComfyUI/input/silent.wav" --seconds 30

Exit codes:
    0  File written.
    2  Bad arguments.

stdlib only, no third-party dependencies.
"""
from __future__ import annotations
import argparse
import sys
import wave
from pathlib import Path


def make_silent_wav(
    path: Path, seconds: float, rate: int, channels: int, sampwidth: int = 2
) -> int:
    """Write `seconds` of silence to `path`. Returns the byte count written."""
    frames = int(round(seconds * rate))
    data = b"\x00" * (frames * channels * sampwidth)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(data)
    return len(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "outfile",
        nargs="?",
        default="silent.wav",
        help="Output path (default: silent.wav in the current folder).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Duration in seconds; make it >= your clip length (default: 10).",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100 — what LTX expects).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Channel count: 2=stereo, 1=mono (default: 2).",
    )
    args = parser.parse_args(argv)

    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.channels not in (1, 2):
        parser.error("--channels must be 1 or 2")

    out = Path(args.outfile)
    if out.parent and not out.parent.exists():
        parser.error(f"output directory does not exist: {out.parent}")

    written = make_silent_wav(out, args.seconds, args.rate, args.channels)
    print(
        f"Wrote {out} — {args.seconds:g}s, {args.rate} Hz, "
        f"{args.channels}ch, 16-bit ({written:,} bytes of silence)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
