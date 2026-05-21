#!/usr/bin/env python3
"""Smoke-test Stage-1 FAESM forward/backward with or without flash-attention."""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_stage1_scisor_style_streaming import (  # noqa: E402
    DEFAULT_P0,
    DEFAULT_SCISOR_CKPT,
    FAESMBaseNoFA,
    SCISORCollator,
    load_stage1_model,
)


def make_item(sequence, insert="GGSG"):
    midpoint = len(sequence) // 2
    corrupted = sequence[:midpoint] + insert + sequence[midpoint:]
    return {
        "seq_id": "smoke",
        "original_sequence": sequence,
        "corrupted_sequence": corrupted,
        "delete_labels": [0.0] * midpoint + [1.0] * len(insert) + [0.0] * (len(sequence) - midpoint),
        "budget_ratio": 0.1,
        "inserted_total_length": float(len(insert)),
        "original_length": len(sequence),
        "corrupted_length": len(corrupted),
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke-test Stage-1 FAESM forward/backward.")
    parser.add_argument("--checkpoint", default=DEFAULT_SCISOR_CKPT)
    parser.add_argument("--p0", default=DEFAULT_P0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-fa", action="store_true")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    if args.disable_fa:
        import SCISOR.continuous_time_diffusion as ctd

        ctd.FAESM_Base = FAESMBaseNoFA

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequence = (alphabet * ((args.length // len(alphabet)) + 1))[: args.length]

    model = load_stage1_model(args, device)
    model.train(True)
    collator = SCISORCollator(
        model.tokenizer.pad_token_id,
        model.tokenizer,
        use_fast_tokenizer=True,
        tokenizer_cache_size=16,
        batch_alignments=True,
    )
    batch = collator([make_item(sequence) for _ in range(args.batch_size)])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    x0 = batch["x0"].long()
    x_t = batch["x_t"].long()
    t = batch["budget_ratio"].squeeze(-1).float().clamp(0.05, 0.95)
    s_values = batch["inserted_total_length"].float()
    log_alignments = batch["log_alignments"].float()

    loss, info = model.forward_preprocessed(x0, t, s_values, x_t, log_alignments)
    loss.backward()
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(
        "OK disable_fa={} device={} loss={:.6f} x0_shape={} xt_shape={} info={}".format(
            args.disable_fa,
            device,
            float(loss.detach().cpu()),
            tuple(x0.shape),
            tuple(x_t.shape),
            info,
        )
    )


if __name__ == "__main__":
    main()
