#!/usr/bin/env python3
"""Convert a local Llama safetensors snapshot to SPARK's row-major FP32 format."""

import argparse
import array
import json
import math
import mmap
import re
import struct
import sys
import tempfile
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
REVISION = "9213176726f574b556790deb65791e0c5aa438b6"


def read_config(source):
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (
        config["model_type"] != "llama"
        or config["hidden_act"] != "silu"
        or config.get("attention_bias", False)
        or config.get("mlp_bias", False)
    ):
        raise ValueError("expected a bias-free Llama decoder with SwiGLU")
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    hidden = config["hidden_size"]
    head_dim = config.get("head_dim", hidden // heads)
    if min(heads, kv_heads, hidden, head_dim) < 1 or (
        head_dim % 2 or heads % kv_heads or hidden != heads * head_dim
    ):
        raise ValueError("invalid grouped-query dimensions")
    rope = config["rope_scaling"]
    if rope["rope_type"] != "llama3":
        raise ValueError("expected Llama 3 rotary scaling")
    return {**config, "head_dim": head_dim}


def expected_shapes(config):
    hidden = config["hidden_size"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    shapes = {
        "model.embed_tokens.weight": [vocab, hidden],
        "model.norm.weight": [hidden],
        "lm_head.weight": [vocab, hidden],
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        shapes.update({
            prefix + "input_layernorm.weight": [hidden],
            prefix + "post_attention_layernorm.weight": [hidden],
            prefix + "self_attn.q_proj.weight": [hidden, hidden],
            prefix + "self_attn.k_proj.weight": [kv_width, hidden],
            prefix + "self_attn.v_proj.weight": [kv_width, hidden],
            prefix + "self_attn.o_proj.weight": [hidden, hidden],
            prefix + "mlp.gate_proj.weight": [intermediate, hidden],
            prefix + "mlp.up_proj.weight": [intermediate, hidden],
            prefix + "mlp.down_proj.weight": [hidden, intermediate],
        })
    return shapes


def fp32_bytes(raw, dtype):
    if dtype == "F32":
        return raw
    if dtype == "BF16":
        result = bytearray(len(raw) * 2)
        result[2::4] = raw[::2]
        result[3::4] = raw[1::2]
        return result
    result = array.array("f", (value for (value,) in struct.iter_unpack("<e", raw)))
    if sys.byteorder != "little":
        result.byteswap()
    return result


def write_weights(source, destination, config):
    shapes = expected_shapes(config)
    tied = config.get("tie_word_embeddings", False)
    tensors = {}
    paths = sorted(source.glob("*.safetensors"))
    if not paths:
        raise ValueError("no safetensors shards in the source directory")
    with (destination / "weights.bin").open("wb") as output:
        for path in paths:
            with path.open("rb") as shard, mmap.mmap(shard.fileno(), 0, access=mmap.ACCESS_READ) as data:
                header_size, = struct.unpack_from("<Q", data)
                payload = 8 + header_size
                if payload > len(data) or header_size > 100_000_000:
                    raise ValueError(f"invalid safetensors header in {path.name}")
                header = json.loads(data[8:payload])
                for name, entry in header.items():
                    if name == "__metadata__":
                        continue
                    if name not in shapes or name in tensors:
                        raise ValueError(f"unexpected or repeated tensor {name}")
                    if entry["shape"] != shapes[name]:
                        raise ValueError(f"unexpected shape for {name}")
                    begin, end = entry["data_offsets"]
                    dtype = entry["dtype"]
                    item_size = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
                    count = math.prod(shapes[name])
                    if (
                        item_size is None or begin < 0 or end < begin or payload + end > len(data)
                        or end - begin != count * item_size
                    ):
                        raise ValueError(f"invalid tensor storage for {name}")
                    tensors[name] = {
                        "shape": shapes[name], "dtype": "F32", "offset": output.tell(),
                    }
                    block_size = 1 << 20
                    for offset in range(payload + begin, payload + end, block_size):
                        raw = data[offset:min(offset + block_size, payload + end)]
                        output.write(fp32_bytes(raw, dtype))
        if tied and "lm_head.weight" not in tensors and "model.embed_tokens.weight" in tensors:
            tensors["lm_head.weight"] = dict(tensors["model.embed_tokens.weight"])
        if missing := shapes.keys() - tensors.keys():
            raise ValueError(f"missing tensors: {', '.join(sorted(missing))}")
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="local HF snapshot")
    parser.add_argument("--output", type=Path, required=True, help="new SPARK model directory")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION, help="full hash of the source snapshot")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full 40-character commit hash")
    if args.source.parent.name == "snapshots" and args.source.name != args.revision:
        parser.error("source snapshot directory disagrees with the specified revision")
    config = read_config(args.source)
    if args.output.exists():
        parser.error("output model directory already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".spark-export-", dir=args.output.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        tensors = write_weights(args.source, staging, config)
        manifest = {
            "format": "spark-f32-v1",
            "model_id": args.model_id,
            "revision": args.revision,
            "config": config,
            "tensors": tensors,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        staging.rename(args.output)


if __name__ == "__main__":
    main()
