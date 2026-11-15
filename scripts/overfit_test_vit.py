from argparse import ArgumentParser
from pathlib import Path

from overfit_cifar100 import main as overfit


def main(run_dir, resume=None, handle_signals=True):
    return overfit(
        model="vit", run_dir=run_dir,
        resume=resume, handle_signals=handle_signals,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_dir", type=Path,
                        help="Directory for a new run; inferred from --resume when resuming")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume training from a checkpoint")
    args = parser.parse_args()
    if args.resume is None and args.run_dir is None:
        parser.error("--run_dir is required when not resuming from a checkpoint")
    main(args.run_dir, resume=args.resume)
