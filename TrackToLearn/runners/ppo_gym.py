#!/usr/bin/env python
import argparse
import json
import torch

from argparse import RawTextHelpFormatter
from os.path import join as pjoin

from TrackToLearn.runners.a2c_train import add_a2c_args
from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.runners.experiment import (
    add_experiment_args,
    add_model_args)
from TrackToLearn.runners.train import (
    add_rl_args)
from TrackToLearn.runners.gym_train import (
    GymTraining,
    add_environment_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class PPOGymTraining(GymTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        env_name: str,
        # RL params
        max_ep: int,
        log_interval: int,
        action_std: float,
        lr: float,
        gamma: float,
        # PPO Params
        lmbda: float,
        entropy_loss_coeff: float,
        eps_clip: int,
        n_update: int,
        K_epochs: int,
        # Model params
        n_latent_var: int,
        hidden_layers: int,
        # Experiment params
        use_gpu: bool,
        rng_seed: int,
        render: bool,
    ):
        """
        Parameters
        ----------
        dataset_file: str
            Path to the file containing the signal data
        subject_id: str
            Subject being trained on (in the signal data)
        seeding_file: str
            Path to the mask where seeds can be generated
        tracking_file: str
            Path to the mask where tracking can happen
        ground_truth_folder: str
            Path to reference streamlines that can be used for
            jumpstarting seeds
        max_ep: int
            How many episodes to run the training.
            An episode corresponds to tracking two-ways on one seed and
            training along the way
        log_interval: int
            Interval at which a test run is done
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        lmbda: float
            Lambda parameter for Generalized Advantage Estimation (GAE):
            John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan:
            “High-Dimensional Continuous Control Using Generalized
             Advantage Estimation”, 2015;
            http://arxiv.org/abs/1506.02438 arXiv:1506.02438
        K_epochs: int
            How many epochs to run the optimizer using the current samples
            PPO allows for many training runs on the same samples
        eps_clip: float
            Clipping parameter for PPO
        rng_seed: int
            Seed for general randomness
        entropy_loss_coeff: float,
            Loss coefficient on policy entropy
            Should sum to 1 with other loss coefficients
        n_seeds_per_voxel: int
            How many seeds to generate per voxel
        max_angle: float
            Maximum angle for tracking
        min_length: int
            Minimum length for streamlines
        max_length: int
            Maximum length for streamlines
        step_size: float
            Step size for tracking
        alignment_weighting: float
            Reward coefficient for alignment with local odfs peaks
        straightness_weighting: float
            Reward coefficient for streamline straightness
        length_weighting: float
            Reward coefficient for streamline length
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        tracking_batch_size: int
            Batch size for tracking during test
        n_latent_var: int
            Width of the NN layers
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        # Experiment params
        use_comet: bool
            Use comet for displaying stats during training
        render: bool
            Render tracking
        run_tractometer: bool
            Run tractometer during validation to see how it's
            doing w.r.t. ground truth data
        use_gpu: bool,
            Use GPU for processing
        rng_seed: int
            Seed for general randomness
        load_teacher: str
            Path to pretrained model for imitation learning
        load_policy: str
            Path to pretrained policy
        """

        super().__init__(
            # Dataset params
            path,
            experiment,
            name,
            env_name,
            # PPO params
            max_ep,
            log_interval,
            action_std,
            lr,
            gamma,
            # Model params
            n_latent_var,
            hidden_layers,
            # Experiment params
            use_gpu,
            rng_seed,
            render,
        )

        self.lmbda = lmbda
        self.entropy_loss_coeff = entropy_loss_coeff
        self.eps_clip = eps_clip
        self.n_update = n_update
        self.K_epochs = K_epochs

    def save_hyperparameters(self):
        self.hyperparameters = {
            # RL parameters
            'id': self.name,
            'experiment': self.experiment,
            'algorithm': 'PPO',
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'action_std': self.action_std,
            'lr': self.lr,
            'gamma': self.gamma,
            # Data parameters
            'input_size': self.input_size,
            # Model parameters
            'experiment_path': self.experiment_path,
            'use_gpu': self.use_gpu,
            'hidden_size': self.n_latent_var,
            'hidden_layers': self.hidden_layers,
            'last_episode': self.last_episode,
            # PPO parameters
            'lmbda': self.lmbda,
            'eps_clip': self.eps_clip,
            'entropy_loss_coeff': self.entropy_loss_coeff
        }

        directory = pjoin(self.experiment_path, "model")
        with open(
            pjoin(directory, "hyperparameters.json"),
            'w'
        ) as json_file:
            json_file.write(
                json.dumps(
                    self.hyperparameters,
                    indent=4,
                    separators=(',', ': ')))

    def get_alg(self):
        # The RL training algorithm
        alg = PPO(
            self.input_size,
            self.action_size,
            self.n_latent_var,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.K_epochs,
            self.n_update,
            self.entropy_loss_coeff,
            self.eps_clip,
            self.n_trajectories,
            False,
            self.rng,
            device)
        return alg


def add_ppo_args(parser):
    parser.add_argument('--eps_clip', default=0.001, type=float,
                        help='Clipping parameter for PPO')
    parser.add_argument('--K_epochs', default=1, type=int,
                        help='Train the model for K epochs')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)

    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)

    add_a2c_args(parser)
    add_ppo_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # Finally, get experiments, and train your models:
    ppo_experiment = PPOGymTraining(
        # Dataset params
        args.path,
        args.experiment,
        args.name,
        args.env_name,
        # RL params
        args.max_ep,
        args.log_interval,
        args.action_std,
        # RL Params
        args.lr,
        args.gamma,
        # PPO Params
        args.lmbda,
        args.entropy_loss_coeff,
        args.eps_clip,
        args.n_update,
        args.K_epochs,
        # Model params
        args.n_latent_var,
        args.hidden_layers,
        # Experiment params
        args.use_gpu,
        args.rng_seed,
        args.render,
    )
    ppo_experiment.run()


if __name__ == '__main__':
    main()
