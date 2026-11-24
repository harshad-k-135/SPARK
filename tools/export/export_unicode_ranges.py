#!/usr/bin/env python3
"""Export the host Unicode categories used by Qwen3.5 byte BPE."""

import itertools
import sys
import unicodedata
from pathlib import Path


def kind(value):
    category = unicodedata.category(chr(value))[0]
    if category in "LNM":
        return {"L": 1, "N": 2, "M": 4}[category]
    if category == "Z" or value in {9, 10, 11, 12, 13, 0x85}:
        return 3
    return 0


def main():
    ranges = []
    for group, values in itertools.groupby(range(0x110000), key=kind):
        first = last = next(values)
        for last in values:
            pass
        if group:
            ranges.append(f"{{0x{first:X}, 0x{last:X}, {group}}}")
    lines = [", ".join(ranges[i:i + 3]) for i in range(0, len(ranges), 3)]
    content = (
        f"// Unicode {unicodedata.unidata_version}. See tools/export/export_unicode_ranges.py.\n"
        + ",\n".join(lines) + ",\n"
    )
    Path(sys.argv[1]).write_text(content, encoding="ascii")


if __name__ == "__main__":
    main()
