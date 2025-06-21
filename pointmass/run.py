import argparse
from deep_sprl.util.parameter_parser import parse_parameters
import torch
import config


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument(
        "--type",
        type=str,
        default="wasserstein",
        choices=[
            "default",
            "random",
            "self_paced",
            "wasserstein",
            "alp_gmm",
            "goal_gan",
            "acl",
            "plr",
            "vds",
            "cp_drl",
        ],
    )
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument(
        "--env", type=str, default="point_mass_2d", choices=["point_mass_2d", "maze"]
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--t_scale", type=float, default=0.0)
    parser.add_argument("--r_scale", type=float, default=0.0)
    parser.add_argument("--s_scale", type=float, default=0.0)
    parser.add_argument("--a_scale", type=float, default=0.0)
    parser.add_argument(
        "--aligned", type=str, default="False", choices=["True", "False"]
    )
    parser.add_argument("--n_cores", type=int, default=1)

    args, remainder = parser.parse_known_args()

    args.aligned = args.aligned == "True"
    config.CPDRL.REWARD_DISAGREEMENT_SCALE = args.r_scale
    config.CPDRL.STATE_DISAGREEMENT_SCALE = args.s_scale
    config.CPDRL.ACTION_DISAGREEMENT_SCALE = args.a_scale
    config.CPDRL.TRANSITION_DISAGREEMENT_SCALE = args.t_scale
    config.CPDRL.ALIGNED = args.aligned

    cp_drl_config = config.CPDRL.to_dict()
    cp_drl_config["env"] = args.env
    cp_drl_config["type"] = args.type
    cp_drl_config["learner"] = args.learner
    cp_drl_config["seed"] = args.seed

    print(f"Running experiment with config: {cp_drl_config}")

    parameters = parse_parameters(remainder)

    torch.set_num_threads(args.n_cores)

    if args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment

        exp = PointMass2DExperiment(
            args.base_log_dir, args.type, args.learner, parameters, args.seed
        )
    elif args.env == "maze":
        from deep_sprl.experiments import MazeExperiment

        exp = MazeExperiment(
            args.base_log_dir, args.type, args.learner, parameters, args.seed
        )
    else:
        raise RuntimeError("Unknown environment '%s'!" % args.env)

    exp.train()
    exp.evaluate()


if __name__ == "__main__":
    main()
