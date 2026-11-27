"""Command-line entry for local model training and evaluation."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import math
import torch
import yaml


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Multi-objective metro expansion")
    parser.add_argument("--weights", nargs=3, type=float, required=True, metavar=("w_G", "w_J", "w_F"))
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--city", default="beijing")
    parser.add_argument("--initial-lines", type=int, default=4)
    parser.add_argument("--budget", type=float, default=40.)
    parser.add_argument("--max-lines", type=int, default=3)
    parser.add_argument("--min-stations", type=int, default=10)
    parser.add_argument("--config", type=Path, default=root / "metro" / "cfg" / "morl.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--mini-batch-size", type=int)
    parser.add_argument("--num-optim-epoch", type=int)
    parser.add_argument("--curriculum-power", type=float, default=3.)
    parser.add_argument("--check", action="store_true", help="Read-only correctness checks without training.")
    args = parser.parse_args()
    if args.seed < 0 or args.threads < 1 or args.rollout_steps < 1:
        parser.error("Seed must be nonnegative; threads and rollout length must be positive.")
    if not math.isfinite(args.curriculum_power) or args.curriculum_power < 1:
        parser.error("Curriculum power must be finite and at least one.")
    for field in ("mini_batch_size", "num_optim_epoch"):
        if getattr(args, field) is not None and getattr(args, field) < 1:
            parser.error("Minibatch size and optimization epochs must be positive.")
    if args.mode == "evaluate" and args.checkpoint is None and not args.check:
        parser.error("Evaluation requires an explicit --checkpoint.")
    if args.mode == "train" and args.checkpoint is not None:
        parser.error("Training always starts from a new model; checkpoint loading is evaluation-only.")
    if not args.check and args.output is None:
        parser.error("Supply an empty --output directory outside the source tree.")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(args.config.read_text())
    from ours.trainer import run, self_check
    if args.check:
        self_check(args, config)
    else:
        run(args, config)


if __name__ == "__main__":
    main()
