import argparse

import hyper_params as hp
from iteration import run_iterations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SourGrape phoneme pretraining and trajectory training."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=hp.iterations,
        help="Set the number of full multi-condition runs to execute.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=hp.generations,
        help="Set the number of generations to run for each condition.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "pretrain", "train"],
        default=hp.stage,
        help="Select which stage to run.",
    )
    parser.add_argument(
        "--model-type",
        choices=["lstm", "seq2seq"],
        default=None,
        help="Override the trajectory model type.",
    )
    parser.add_argument(
        "--penalty-loss-type",
        choices=["sigmoid_bce", "relu_mse", "softplus_mse"],
        default=None,
        help="Override the auxiliary penalty loss type.",
    )
    parser.add_argument(
        "--penalty-loss-weight",
        type=float,
        default=None,
        help="Override the auxiliary penalty loss weight.",
    )
    return parser.parse_args()


def override_hyperparams(args: argparse.Namespace) -> None:
    hp.iterations = args.iterations
    hp.generations = args.generations
    hp.stage = args.stage
    if args.model_type is not None:
        hp.model_type = args.model_type
    if args.penalty_loss_type is not None:
        hp.penalty_loss_type = args.penalty_loss_type
    if args.penalty_loss_weight is not None:
        hp.penalty_loss_weight = args.penalty_loss_weight


def main() -> None:
    args = parse_args()
    override_hyperparams(args)
    run_iterations(
        seed=hp.seed,
        num_iterations=hp.iterations,
        num_generations=hp.generations,
        stage=hp.stage,
    )


if __name__ == "__main__":
    main()
