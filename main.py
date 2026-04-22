import argparse

from iteration import run_generations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SourGrape phoneme pretraining and trajectory training."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set the base random seed for the run.",
    )
    parser.add_argument(
        "--condition",
        choices=["glide", "fricative", "all"],
        default="all",
        help="Select which condition to run.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Set the number of generations to run for each condition.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "pretrain", "train"],
        default="all",
        help="Select which stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.condition == "all":
        conditions = ["glide", "fricative"]
    else:
        conditions = [args.condition]

    for condition in conditions:
        run_generations(
            seed=args.seed,
            condition=condition,
            num_generations=args.generations,
            stage=args.stage,
        )


if __name__ == "__main__":
    main()
