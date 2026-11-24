#!/usr/bin/env python3
"""Compile Llama 3 tokenizer.json into byte vocabulary and BPE merge tables."""

import argparse
import itertools
import json
import struct
import tempfile
import unicodedata
from pathlib import Path

LLAMA_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def byte_alphabet():
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    alphabet = {chr(value): value for value in direct}
    alphabet.update({
        chr(256 + index): value for index, value in enumerate(sorted(set(range(256)) - direct))
    })
    return alphabet


def unicode_class(value):
    character = chr(value)
    category = unicodedata.category(character)
    if category.startswith("L"):
        return 1
    if category.startswith("N"):
        return 2
    if category.startswith("Z") or value in (9, 10, 11, 12, 13, 0x85):
        return 3
    return 0


def unicode_ranges():
    ranges = []
    for kind, values in itertools.groupby(range(0x110000), key=unicode_class):
        first = last = next(values)
        for last in values:
            pass
        if kind:
            ranges.append((first, last, kind))
    return ranges


def export_tokenizer(source, destination):
    data = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    model = data["model"]
    stages = data["pre_tokenizer"].get("pretokenizers", [])
    if (
        data.get("normalizer") is not None or model["type"] != "BPE"
        or model.get("dropout") or model.get("unk_token") is not None
        or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix")
        or len(stages) != 2 or stages[0]["type"] != "Split"
        or stages[0]["pattern"].get("Regex") != LLAMA_SPLIT
        or stages[0]["behavior"] != "Isolated" or stages[0]["invert"]
        or stages[1]["type"] != "ByteLevel" or stages[1]["add_prefix_space"]
        or stages[1]["use_regex"] or data["decoder"]["type"] != "ByteLevel"
    ):
        raise ValueError("unsupported Llama 3 tokenizer pre-tokenization or BPE format")
    alphabet = byte_alphabet()
    vocab = model["vocab"]
    entries = {token_id: (bytes(alphabet[c] for c in text), 0) for text, token_id in vocab.items()}
    for token in data["added_tokens"]:
        if (
            not token["special"] or token["lstrip"] or token["rstrip"] or token["single_word"]
            or not token["content"].startswith("<|") or token["id"] in entries
        ):
            raise ValueError(f"unsupported added token: {token['content']}")
        entries[token["id"]] = token["content"].encode("utf-8"), 1
    count = len(entries)
    if set(entries) != set(range(count)):
        raise ValueError("expected contiguous vocabulary IDs")
    merges = []
    for pair in model["merges"]:
        left, right = pair.split(" ") if isinstance(pair, str) else pair
        merges.append((vocab[left], vocab[right], vocab[left + right]))
    ranges = unicode_ranges()
    with destination.open("wb") as output:
        output.write(b"SPKTT01\n")
        output.write(struct.pack("<4I", count, len(merges), len(ranges), bool(model.get("ignore_merges"))))
        for token_id in range(count):
            piece, special = entries[token_id]
            output.write(struct.pack("<2I", len(piece), special))
            output.write(piece)
        for entry in itertools.chain(merges, ranges):
            output.write(struct.pack("<3I", *entry))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="local HF snapshot")
    parser.add_argument("--output", type=Path, required=True, help="existing SPARK model directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix=".spark-tokenizer-", dir=args.output) as temporary:
        table = Path(temporary) / "tokenizer.bin"
        export_tokenizer(args.source, table)
        table.replace(args.output / table.name)


if __name__ == "__main__":
    main()
