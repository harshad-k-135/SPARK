#!/usr/bin/env python3
"""Build an offline synthetic checkpoint and verify the compiled CPU engine."""

import argparse
import json
import math
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/export"))
from export_hf_to_spark import expected_shapes
from export_tokenizer import LLAMA_SPLIT, byte_alphabet

BOS = 258
EOS = 259
REVISION = "0" * 40
CONFIG = {
    "model_type": "llama", "hidden_act": "silu", "head_dim": 4, "hidden_size": 8,
    "intermediate_size": 12, "num_attention_heads": 2, "num_key_value_heads": 1,
    "num_hidden_layers": 2, "vocab_size": 260, "max_position_embeddings": 128,
    "bos_token_id": BOS, "eos_token_id": [EOS], "tie_word_embeddings": True,
    "rms_norm_eps": 1e-5, "rope_theta": 64,
    "rope_scaling": {
        "rope_type": "llama3", "factor": 4, "low_freq_factor": 1,
        "high_freq_factor": 4, "original_max_position_embeddings": 32,
    },
}
CASES = (
    ("spaces", "  hi hi", [32, 257, 257]),
    ("utf8", " hi\u00e9\t\n!", [257, 195, 169, 9, 10, 33]),
    ("special", "hi<|end|>hi", [256, EOS, 256]),
)


def checkpoint(source):
    source.mkdir()
    (source / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    alphabet = {byte: char for char, byte in byte_alphabet().items()}
    vocab = {char: byte for byte, char in alphabet.items()}
    vocab.update({"hi": 256, alphabet[32] + "hi": 257})
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [["h", "i"], [alphabet[32], "hi"]]},
        "pre_tokenizer": {"pretokenizers": [
            {"type": "Split", "pattern": {"Regex": LLAMA_SPLIT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
        ]},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": token, "special": True, "lstrip": False, "rstrip": False,
             "single_word": False, "content": content}
            for token, content in ((BOS, "<|begin|>"), (EOS, "<|end|>"))
        ],
    }
    (source / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    rng = random.Random(1247)
    header = {}
    payload = bytearray()
    weights = {}
    for name, shape in expected_shapes(CONFIG).items():
        if name == "lm_head.weight":
            continue
        values = [
            (1 if "norm" in name else 0) + rng.randint(-19, 19) / 64
            for _ in range(math.prod(shape))
        ]
        dtype = ("F32", "F16", "BF16")[len(header) % 3]
        raw = struct.pack(f"<{len(values)}{'e' if dtype == 'F16' else 'f'}", *values)
        if dtype == "BF16":
            raw = bytes(byte for index, byte in enumerate(raw) if index % 4 >= 2)
        header[name] = {
            "shape": shape, "dtype": dtype, "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
        weights[name] = (values if len(shape) == 1 else
                         [values[i:i + shape[1]] for i in range(0, len(values), shape[1])])
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    return weights


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def norm(vector, scale):
    inverse = 1 / math.sqrt(dot(vector, vector) / len(vector) + CONFIG["rms_norm_eps"])
    return [inverse * value * weight for value, weight in zip(vector, scale, strict=True)]


def project(vector, matrix):
    return [dot(row, vector) for row in matrix]


def rope(vector, position):
    # Raw wavelengths are 2*pi and 16*pi. The first frequency stays
    # unchanged and the second is divided by 4 under this fixture's scale.
    out = vector.copy()
    for start in range(0, len(vector), 4):
        for i, frequency in enumerate((1, 1 / 32)):
            angle = position * frequency
            a, b = vector[start + i], vector[start + i + 2]
            out[start + i] = a * math.cos(angle) - b * math.sin(angle)
            out[start + i + 2] = a * math.sin(angle) + b * math.cos(angle)
    return out


def logits(weights, tokens):
    embedding = weights["model.embed_tokens.weight"]
    hidden = [embedding[token] for token in tokens]
    for index in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        layer = {name[len(prefix):]: value for name, value in weights.items() if name.startswith(prefix)}
        normalized = [norm(h, layer["input_layernorm.weight"]) for h in hidden]
        keys = [rope(project(h, layer["self_attn.k_proj.weight"]), p) for p, h in enumerate(normalized)]
        values = [project(h, layer["self_attn.v_proj.weight"]) for h in normalized]
        for position, h in enumerate(normalized):
            query = rope(project(h, layer["self_attn.q_proj.weight"]), position)
            attention = []
            for head in (query[:4], query[4:]):
                numerators = [math.exp(dot(head, key) / 2) for key in keys[:position + 1]]
                total = math.fsum(numerators)
                attention.extend(
                    dot(numerators, column) / total for column in zip(*values[:position + 1])
                )
            residual = [a + b for a, b in zip(
                hidden[position], project(attention, layer["self_attn.o_proj.weight"]), strict=True,
            )]
            mlp = norm(residual, layer["post_attention_layernorm.weight"])
            gate = project(mlp, layer["mlp.gate_proj.weight"])
            up = project(mlp, layer["mlp.up_proj.weight"])
            activated = [a / (1 + math.exp(-a)) * b for a, b in zip(gate, up, strict=True)]
            down = project(activated, layer["mlp.down_proj.weight"])
            hidden[position] = [a + b for a, b in zip(residual, down, strict=True)]
    return project(norm(hidden[-1], weights["model.norm.weight"]), embedding)


def make_reference(weights, path):
    entries = []
    for name, prompt, tokens in CASES:
        tokens = [BOS, *tokens]
        output = []
        first = logits(weights, tokens)
        current = first
        for _ in range(8):
            output.append(max(range(len(current)), key=current.__getitem__))
            current = logits(weights, tokens + output)
        entries.append({"id": name, "prompt": prompt, "prompt_ids": tokens,
                        "generated_ids": output, "first_logits": first})
    path.write_text(json.dumps({
        "format": "spark-golden-v1", "dtype": "float32", "model_id": "synthetic",
        "revision": REVISION, "cases": entries,
    }), encoding="utf-8")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        source, model, reference = root / "hf", root / "spark", root / "reference.json"
        weights = checkpoint(source)
        entries = make_reference(weights, reference)
        for script, extra in (
            ("export_hf_to_spark.py", ["--model-id", "synthetic", "--revision", REVISION]),
            ("export_tokenizer.py", []),
        ):
            subprocess.run([
                sys.executable, ROOT / "tools/export" / script, "--source", source,
                "--output", model, *extra,
            ], check=True)
        subprocess.run([args.check.resolve(), "--smoke", model, reference], check=True)

        command = [args.cli.resolve(), "generate", "--model", model, "--context", "128",
                   "--prompt", entries[0]["prompt"], "--max-new-tokens", "8"]
        expected = bytearray()
        for token in entries[0]["generated_ids"]:
            if token == EOS:
                break
            expected.extend(bytes([token]) if token < 256 else
                            {256: b"hi", 257: b" hi", BOS: b""}[token])
        greedy = subprocess.run(command, capture_output=True, check=True)
        if greedy.stdout != expected + b"\n":
            raise RuntimeError("generation CLI did not produce reference greedy bytes")
        sampling = ["--temperature", "0.8", "--top-p", "0.9", "--seed", "42"]
        first = subprocess.run(command + sampling, capture_output=True, check=True)
        second = subprocess.run(command + sampling, capture_output=True, check=True)
        if first.stdout != second.stdout:
            raise RuntimeError("fixed-seed sampling is not repeatable")
        overflow = subprocess.run(command + ["--context", "8"], capture_output=True)
        if overflow.returncode == 0 or b"exceeds the context limit" not in overflow.stderr:
            raise RuntimeError("generation accepted an over-budget request")
        print('{"event":"cli_smoke","greedy_bytes":true,"seed_repeatability":true,"context_rejection":true}')


if __name__ == "__main__":
    main()
