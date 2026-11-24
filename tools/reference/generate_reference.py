#!/usr/bin/env python3
"""Generate FP32 CPU reference logits and fixed-length continuations."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
STEPS = 64


def fixture_prompt(case, tokenizer):
    text = case["text"]
    ids = tokenizer.encode(text, add_special_tokens=True)
    target = case.get("tokens")
    if target is not None:
        if len(ids) < target:
            # Each repeated paragraph ends with punctuation and starts on a new line.
            paragraph = text.rsplit("\n\n", 1)[-1]
            text += ("\n\n" + paragraph) * (target // max(1, len(ids) // 2) + 2)
            ids = tokenizer.encode(text, add_special_tokens=True)
        ids = ids[:target]
        text = tokenizer.decode(ids[1:], clean_up_tokenization_spaces=False)
        if text.endswith("\ufffd") or tokenizer.encode(text, add_special_tokens=True) != ids:
            raise ValueError(f"{case['id']} does not have a stable {target}-token fixture prefix")
    if not 16 <= len(ids) <= 1500 or ids[0] != tokenizer.bos_token_id:
        raise ValueError(f"{case['id']} does not have a valid prompt length or BOS")
    decoded = tokenizer.decode(ids[1:], clean_up_tokenization_spaces=False)
    if text != decoded:
        raise ValueError(f"{case['id']} does not have a byte-exact tokenizer round-trip")
    return text, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="local HF snapshot")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--prompts", type=Path, default=ROOT / "tests/golden/prompts.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/golden/reference.json")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full 40-character commit hash")
    if args.source.parent.name == "snapshots" and args.source.name != args.revision:
        parser.error("source snapshot directory disagrees with the specified revision")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True, use_fast=True)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    fixtures = [(case["id"], *fixture_prompt(case, tokenizer)) for case in prompts]
    lengths = [len(ids) for _, _, ids in fixtures]
    if len({name for name, _, _ in fixtures}) < 20 or min(lengths) != 16 or max(lengths) != 1500:
        raise ValueError("expected at least 20 distinct fixtures spanning 16 through 1500 tokens")

    model = AutoModelForCausalLM.from_pretrained(
        args.source, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu").eval()
    results = []
    with torch.inference_mode():
        for name, text, ids in fixtures:
            current = torch.tensor([ids], dtype=torch.long)
            past = None
            entry = {"id": name, "prompt": text, "prompt_ids": ids, "generated_ids": []}
            for step in range(STEPS):
                result = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
                last = result.logits[0, -1]
                if step == 0:
                    entry["first_logits"] = last.tolist()
                current = last.argmax().reshape(1, 1)
                entry["generated_ids"].append(current.item())
                past = result.past_key_values
            results.append(entry)
            print(f"{name}: {len(ids)} prompt + {STEPS} greedy tokens", flush=True)
    output = {
        "format": "spark-golden-v1", "model_id": args.model_id, "revision": args.revision,
        "dtype": "float32", "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "transformers": transformers.__version__, "cases": results,
    }
    pending = args.output.with_suffix(args.output.suffix + ".tmp")
    pending.write_text(
        json.dumps(output, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(args.output)


if __name__ == "__main__":
    main()
