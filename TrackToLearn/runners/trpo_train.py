#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import json
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment
from os.path import join as pjoin

from TrackToLearn.runners.a2c_train import add_a2c_args
from TrackToLearn.algorithms.trpo import TRPO
from TrackToLearn.runners.experiment import (
    add_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.runners.train import (
    add_rl_args,
    TrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TRPOTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        dataset_file: str,
        subject_id: str,
        test_dataset_file: str,
        test_subject_id: str,
        reference_file: str,
        ground_truth_folder: str,
        # RL params
        max_ep: int,
        log_interval: int,
        action_std: float,
        valid_noise: float,
        lr: float,
        gamma: float,
        # TRPO Params
        lmbda: float,
        entropy_loss_coeff: float,
        delta: int,
        max_backtracks: int,
        backtrack_coeff: float,
        n_update: int,
        K_epochs: int,
        # Env params
        n_seeds_per_voxel: int,
        max_angle: float,
        min_length: int,
        max_length: int,
        step_size: float,  # Step size (in mm)
        alignment_weighting: float,
        straightness_weighting: float,
        length_weighting: float,
        target_bonus_factor: float,
        exclude_penalty_factor: float,
        angle_penalty_factor: float,
        tracking_batch_size: int,
        n_signal: int,
        n_dirs: int,
        gm_seeding: bool,
        no_retrack: bool,
        # Model params
        n_latent_var: int,
        hidden_layers: int,
        add_neighborhood: float,
        # Experiment params
        use_gpu: bool,
        rng_seed: int,
        comet_experiment: Experiment,
        render: bool,
        run_tractometer: bool,
        load_teacher: str,
        load_policy: str,
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
            TRPO allows for many training runs on the same samples
        eps_clip: float
            Clipping parameter for TRPO
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
            dataset_file,
            subject_id,
            test_dataset_file,
            test_subject_id,
            reference_file,
            ground_truth_folder,
            # TRPO params
            max_ep,
            log_interval,
            action_std,
            valid_noise,
            lr,
            gamma,
            # Env params
            n_seeds_per_voxel,
            max_angle,
            min_length,
            max_length,
            step_size,  # Step size (in mm)
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            tracking_batch_size,
            n_signal,
            n_dirs,
            gm_seeding,
            no_retrack,
            # Model params
            n_latent_var,
            hidden_layers,
            add_neighborhood,
            # Experiment params
            use_gpu,
            rng_seed,
            comet_experiment,
            render,
            run_tractometer,
            load_teacher,
            load_policy
        )

        self.lmbda = lmbda
        self.entropy_loss_coeff = entropy_loss_coeff
        self.delta = delta
        self.max_backtracks = max_backtracks
        self.backtrack_coeff = backtrack_coeff
        self.n_update = n_update
        self.K_epochs = K_epochs

    def save_hyperparameters(self):
        self.hyperparameters = {
            # RL parameters
            'id': self.name,
            'experiment': self.experiment,
            'algorithm': 'TRPO',
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'action_std': self.action_std,
            'lr': self.lr,
            'gamma': self.gamma,
            # Data parameters
            'input_size': self.input_size,
            'add_neighborhood': self.add_neighborhood,
            'step_size': self.step_size,
            'random_seed': self.rng_seed,
            'dataset_file': self.dataset_file,
            'subject_id': self.subject_id,
            'n_seeds_per_voxel': self.n_seeds_per_voxel,
            'max_angle': self.max_angle,
            'min_length': self.min_length,
            'max_length': self.max_length,
            # Model parameters
            'experiment_path': self.experiment_path,
            'use_gpu': self.use_gpu,
            'hidden_size': self.n_latent_var,
            'hidden_layers': self.hidden_layers,
            'last_episode': self.last_episode,
            'tracking_batch_size': self.tracking_batch_size,
            'n_signal': self.n_signal,
            'n_dirs': self.n_dirs,
            'gm_seeding': self.gm_seeding,
            'no_retrack': self.no_retrack,
            # Reward parameters
            'alignment_weighting': self.alignment_weighting,
            'straightness_weighting': self.straightness_weighting,
            'length_weighting': self.length_weighting,
            'target_bonus_factor': self.target_bonus_factor,
            'exclude_penalty_factor': self.exclude_penalty_factor,
            'angle_penalty_factor': self.angle_penalty_factor,
            # TRPO parameters
            'lmbda': self.lmbda,
            'delta': self.delta,
            'backtrack_coeff': self.backtrack_coeff,
            'max_backtracks': self.max_backtracks,
            'entropy_loss_coeff': self.entropy_loss_coeff,
            'n_update': self.n_update,
            'K_epochs': self.K_epochs,
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
        alg = TRPO(
            self.input_size,
            3,
            self.n_latent_var,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.entropy_loss_coeff,
            self.delta,
            self.max_backtracks,
            self.backtrack_coeff,
            self.n_update,
            self.K_epochs,
            self.tracking_batch_size,
            self.gm_seeding,
            self.rng,
            device)
        return alg


def add_trpo_args(parser):
    parser.add_argument('--max_backtracks', default=10, type=int,
                        help='Backtracks for conjugate gradient')
    parser.add_argument('--delta', default=0.01, type=float,
                        help='Clipping parameter for TRPO')
    parser.add_argument('--backtrack_coeff', default=0.5, type=float,
                        help='Backtracking coefficient')
    parser.add_argument('--K_epochs', default=1, type=int,
                        help='Train the model for K epochs')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)
    add_data_args(parser)

    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_tracking_args(parser)

    add_a2c_args(parser)
    add_trpo_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    experiment = Experiment(project_name=args.experiment,
                            workspace='TrackToLearn', parse_args=False,
                            auto_metric_logging=False,
                            disabled=not args.use_comet)
    print("n_update", args.n_update)
    # Finally, get experiments, and train your models:
    trpo_experiment = TRPOTrackToLearnTraining(
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
        args.action_std,
        args.valid_noise,
        args.lr,
        args.gamma,
        # TRPO params
        args.lmbda,
        args.entropy_loss_coeff,
        args.delta,
        args.max_backtracks,
        args.backtrack_coeff,
        args.n_update,
        args.K_epochs,
        # Env params
        args.n_seeds_per_voxel,
        args.max_angle,
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
    trpo_experiment.run()


if __name__ == '__main__':
    main()
