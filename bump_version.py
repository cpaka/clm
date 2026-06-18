#!/usr/bin/env python3
"""bump_version.py — increment VERSION and corpus size in modal_app.py.

Usage:
    python bump_version.py           # auto-increment patch (v4 → v4.1, v4.1 → v4.2)
    python bump_version.py --dry-run # print what would change without writing

Version series:
    v4    → 1M chars / 5K seqs   (baseline)
    v4.1  → 2M chars / 10K seqs
    v4.2  → 5M chars / 20K seqs
    v4.3  → 10M chars / 40K seqs
    ...   doubling each step
"""
import re
import sys
from pathlib import Path

APP = Path(__file__).parent / "modal_app.py"

# Corpus scaling: each bump doubles chars and seqs
_SCALE_FACTOR = 2


def _parse_version(s: str) -> tuple[int, int]:
    """'v4' → (4,0), 'v4.1' → (4,1), 'v4.2' → (4,2)"""
    m = re.fullmatch(r"v(\d+)(?:\.(\d+))?", s)
    if not m:
        raise ValueError(f"Unrecognised version string: {s!r}")
    return int(m.group(1)), int(m.group(2) or 0)


def _format_version(major: int, minor: int) -> str:
    return f"v{major}.{minor}" if minor else f"v{major}"


def bump(dry_run: bool = False):
    text = APP.read_text()

    # Extract current VERSION
    ver_m = re.search(r'^VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not ver_m:
        sys.exit("ERROR: could not find VERSION in modal_app.py")
    cur_ver = ver_m.group(1)
    major, minor = _parse_version(cur_ver)
    new_ver = _format_version(major, minor + 1)

    # Extract current corpus chars
    chars_m = re.search(r'"max_chars":\s*(\d[\d_]*)', text)
    if not chars_m:
        sys.exit("ERROR: could not find max_chars in modal_app.py")
    cur_chars = int(chars_m.group(1).replace("_", ""))
    new_chars = cur_chars * _SCALE_FACTOR

    # Extract current max_sequences
    seqs_m = re.search(r'"max_sequences":\s*(\d[\d_]*)', text)
    if not seqs_m:
        sys.exit("ERROR: could not find max_sequences in modal_app.py")
    cur_seqs = int(seqs_m.group(1).replace("_", ""))
    new_seqs = cur_seqs * _SCALE_FACTOR

    print(f"VERSION:       {cur_ver!r} → {new_ver!r}")
    print(f"max_chars:     {cur_chars:,} → {new_chars:,}")
    print(f"max_sequences: {cur_seqs:,} → {new_seqs:,}")

    if dry_run:
        print("\n[dry-run] no files modified")
        return

    def _fmt_int(n: int) -> str:
        """Format large ints with underscores for readability."""
        s = str(n)
        return "_".join(s[max(0, i-3):i] for i in range(len(s), 0, -3))[::-1].lstrip("_") if len(s) > 3 else s

    new_text = text
    new_text = new_text.replace(f'VERSION = "{cur_ver}"', f'VERSION = "{new_ver}"', 1)
    new_text = re.sub(
        r'("max_chars":\s*)(\d[\d_]*)',
        lambda m: m.group(1) + _fmt_int(new_chars),
        new_text, count=1,
    )
    new_text = re.sub(
        r'("max_sequences":\s*)(\d[\d_]*)',
        lambda m: m.group(1) + _fmt_int(new_seqs),
        new_text, count=1,
    )

    APP.write_text(new_text)
    print(f"\n✓ modal_app.py updated to {new_ver}")
    print("Next steps:")
    print(f"  make fetch-corpus   # sample {new_chars:,} chars from Wikipedia")
    print(f"  make train-parallel # train {new_seqs:,} seqs on GPU")
    print(f"  make deploy         # publish {new_ver}")


if __name__ == "__main__":
    bump(dry_run="--dry-run" in sys.argv)
