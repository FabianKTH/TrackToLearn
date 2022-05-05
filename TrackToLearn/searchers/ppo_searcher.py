#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from TrackToLearn.runners.ppo_train import (
    parse_args,
    PPOTrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    from comet_ml import Optimizer

    # We only need to specify the algorithm and hyperparameters to use:

    # We only need to specify the algorithm and hyperparameters to use:
    config = {
        # We pick the Bayes algorithm:
        "algorithm": "grid",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "lr": {
                "type": "discrete",
                "values": [0.00005, 0.0001, 0.0005]},
            "gamma": {
                "type": "discrete",
                "values": [0.75, 0.85, 0.9, 0.95, 0.99]},
            "action_std": {
                "type": "discrete",
                "values": [0.0]},
            "entropy_loss_coeff": {
                "type": "discrete",
                "values": [0.00001, 0.0001, 0.001, 0.01, 0.1]},
            "lmbda": {
                "type": "discrete",
                "values": [0.95]},
            "eps_clip": {
                "type": "discrete",
                "values": [0.05, 0.1, 0.2]},
            "K_epochs": {
                "type": "discrete",
                "values": [10, 50, 80]},
            "n_update": {
                "type": "discrete",
                "values": [50]},
            "max_angle": {
                "type": "discrete",
                "values": [100]},
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "Reward",
            "objective": "maximize",
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config, project_name=args.experiment)

    for experiment in opt.get_experiments():
        experiment.auto_metric_logging = False
        experiment.workspace = 'TrackToLearn'
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        lr = experiment.get_parameter("lr")
        gamma = experiment.get_parameter("gamma")
        lmbda = experiment.get_parameter("lmbda")
        entropy_loss_coeff = experiment.get_parameter("entropy_loss_coeff")
        action_std = experiment.get_parameter('action_std')

        K_epochs = experiment.get_parameter("K_epochs")
        n_update = experiment.get_parameter("n_update")
        eps_clip = experiment.get_parameter("eps_clip")

        max_angle = experiment.get_parameter("max_angle")

        ppo_experiment = PPOTrackToLearnTraining(
            # Dataset params
            args.path,
            args.experiment,
            args.name,
            args.dataset_file,
            args.subject_id,
            args.test_dataset_file,
            args.test_subject_id,
            args.reference_file,
            args.ground_truth_folder,
            # RL params
            args.max_ep,
            args.log_interval,
            action_std,
            args.valid_noise,
            lr,
            gamma,
            # PPO params
            lmbda,
            n_update,
            K_epochs,
            eps_clip,
            entropy_loss_coeff,
            # Env params
            args.n_seeds_per_voxel,
            max_angle,
            args.min_length,
            args.max_length,
            args.step_size,  # Step size (in mm)
            args.alignment_weighting,
            args.straightness_weighting,
            args.length_weighting,
            args.target_bonus_factor,
            args.exclude_penalty_factor,
            args.angle_penalty_factor,
            args.tracking_batch_size,
            args.n_signal,
            args.n_dirs,
            args.gm_seeding,
            args.no_retrack,
            # Model params
            args.n_latent_var,
            args.hidden_layers,
            args.add_neighborhood,
            # Experiment params
            args.use_gpu,
            args.rng_seed,
            experiment,
            args.render,
            args.run_tractometer,
            args.load_teacher,
            args.load_policy,
        )
        ppo_experiment.run()


if __name__ == '__main__':
    main()
