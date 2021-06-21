"""Embed out of sample data using npmp architecture.

Attributes:
    MODEL_FEATURES (List): Data saved for all models
    FPS (int): Frame rate of the video
    IMAGE_SIZE (List): Dimensions of images in video
    LSTM_ACTIONS (Dict): Lstm actions and node names
    LSTM_NETWORK_FEATURES (Dict): Lstm data to save
    LSTM_INPUTS (Dict): Lstm inputs and node names
    LSTM_STATES (List): Lstm state features
    mjlib (module): Shorthand for mjbindings.mjlib
    MLP_ACTIONS (Dict): Mlp actions and node names
    MLP_NETWORK_FEATURES (Dict): Mlp data to save
    MLP_INPUTS (Dict): MLP inputs and node names
    MLP_STATES (List): MLP state features
    OBSERVATIONS (List): Model observations
"""
import dm_control
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.walkers import rodent
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control import composer
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import transformations
from dm_control.suite.wrappers import action_scale
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
from scipy.io import loadmat
import pickle
import numpy as np
import imageio
from scipy.io import savemat
import argparse
import ast
import os
import pickle
import yaml
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
from typing import Dict, List, Text, Tuple
import abc
from observer import Observer, MlpObserver, LstmObserver
from feeder import MlpFeeder, LstmFeeder
from system import System
from loop import Loop, OpenLoop, ClosedLoop

mjlib = mjbindings.mjlib

OBSERVATIONS = [
    "walker/actuator_activation",
    "walker/appendages_pos",
    "walker/joints_pos",
    "walker/joints_vel",
    "walker/sensors_accelerometer",
    "walker/sensors_gyro",
    "walker/sensors_touch",
    "walker/sensors_torque",
    "walker/sensors_velocimeter",
    "walker/tendons_pos",
    "walker/tendons_vel",
    "walker/world_zaxis",
    "walker/reference_rel_bodies_pos_local",
    "walker/reference_rel_bodies_quats",
]

MLP_STATES = [
    "latent",
    "target_latent",
    "dummy_core_state",
    "dummy_target_core_state",
    "dummy_policy_state_level_1",
    "dummy_policy_state_level_2",
    "dummy_target_policy_state_level_1",
    "dummy_target_policy_state_level_2",
]

LSTM_STATES = [
    "latent",
    "target_latent",
    "lstm_policy_hidden_level_1",
    "lstm_policy_cell_level_1",
    "lstm_policy_hidden_level_2",
    "lstm_policy_cell_level_2",
]

MLP_INPUTS = {
    "step_type": "step_type_2:0",
    "reward": "reward_2:0",
    "discount": "discount_1:0",
    "walker/actuator_activation": "walker/actuator_activation_1:0",
    "walker/appendages_pos": "walker/appendages_pos_1:0",
    "walker/body_height": "walker/body_height_1:0",
    "walker/end_effectors_pos": "walker/end_effectors_pos_1:0",
    "walker/joints_pos": "walker/joints_pos_1:0",
    "walker/joints_vel": "walker/joints_vel_1:0",
    "walker/sensors_accelerometer": "walker/sensors_accelerometer_1:0",
    "walker/sensors_force": "walker/sensors_force_1:0",
    "walker/sensors_gyro": "walker/sensors_gyro_1:0",
    "walker/sensors_torque": "walker/sensors_torque_1:0",
    "walker/sensors_touch": "walker/sensors_touch_1:0",
    "walker/sensors_velocimeter": "walker/sensors_velocimeter_1:0",
    "walker/tendons_pos": "walker/tendons_pos_1:0",
    "walker/tendons_vel": "walker/tendons_vel_1:0",
    "walker/world_zaxis": "walker/world_zaxis_1:0",
    "walker/reference_rel_bodies_pos_local": "walker/reference_rel_bodies_pos_local_1:0",
    "walker/reference_rel_bodies_quats": "walker/reference_rel_bodies_quats_1:0",
    "dummy_core_state": "state_9:0",
    "dummy_target_core_state": "state_10:0",
    "dummy_policy_state_level_1": "state_11:0",
    "dummy_policy_state_level_2": "state_12:0",
    "dummy_target_policy_state_level_1": "state_14:0",
    "dummy_target_policy_state_level_2": "state_15:0",
    "latent": "state_13:0",
    "target_latent": "state_16:0",
}

LSTM_INPUTS = {
    "step_type": "step_type_2:0",
    "reward": "reward_2:0",
    "discount": "discount_1:0",
    "walker/actuator_activation": "walker/actuator_activation_1:0",
    "walker/appendages_pos": "walker/appendages_pos_1:0",
    "walker/body_height": "walker/body_height_1:0",
    "walker/end_effectors_pos": "walker/end_effectors_pos_1:0",
    "walker/joints_pos": "walker/joints_pos_1:0",
    "walker/joints_vel": "walker/joints_vel_1:0",
    "walker/sensors_accelerometer": "walker/sensors_accelerometer_1:0",
    "walker/sensors_force": "walker/sensors_force_1:0",
    "walker/sensors_gyro": "walker/sensors_gyro_1:0",
    "walker/sensors_torque": "walker/sensors_torque_1:0",
    "walker/sensors_touch": "walker/sensors_touch_1:0",
    "walker/sensors_velocimeter": "walker/sensors_velocimeter_1:0",
    "walker/tendons_pos": "walker/tendons_pos_1:0",
    "walker/tendons_vel": "walker/tendons_vel_1:0",
    "walker/world_zaxis": "walker/world_zaxis_1:0",
    "walker/reference_rel_bodies_pos_local": "walker/reference_rel_bodies_pos_local_1:0",
    "walker/reference_rel_bodies_quats": "walker/reference_rel_bodies_quats_1:0",
    "lstm_policy_hidden_level_1": "state_22:0",
    "lstm_policy_cell_level_1": "state_23:0",
    "lstm_policy_hidden_level_2": "state_24:0",
    "lstm_policy_cell_level_2": "state_25:0",
    "latent": "state_26:0",
    "target_latent": "state_32:0",
}

MLP_ACTIONS = {
    "action": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag/sample/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_chain_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_scale_matvec_linear_operator/forward/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift/forward/add:0",
    "dummy_core_state": "agent_0/step_1/reset_core/Select:0",
    "dummy_target_core_state": "agent_0/step_1/reset_core_2/Select:0",
    "dummy_policy_state_level_1": "agent_0/step_1/reset_core_1/Select:0",
    "dummy_policy_state_level_2": "agent_0/step_1/reset_core_1/Select_1:0",
    "dummy_target_policy_state_level_1": "agent_0/step_1/reset_core_1_1/Select:0",
    "dummy_target_policy_state_level_2": "agent_0/step_1/reset_core_1_1/Select_1:0",
    "latent": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "latent_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "target_latent": "agent_0/step_1/reset_core_1_1/MultiLevelSamplerWithARPrior/add_2:0",
    "prior_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/mul_1:0",
    "level_1_scale": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add:0",
    "level_1_loc": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "latent_sample": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "action_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0",
}

LSTM_ACTIONS = {
    "action": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag/sample/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_chain_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_scale_matvec_linear_operator/forward/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift/forward/add:0",
    "action_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0",
    "lstm_policy_hidden_level_1": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm/mul_2:0",
    "lstm_policy_cell_level_1": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm/add_2:0",
    "lstm_policy_hidden_level_2": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm_1/mul_2:0",
    "lstm_policy_cell_level_2": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_torso/deep_rnn/lstm_1/add_2:0",
    "latent": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "latent_mean": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "latent_sample": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0",
    "level_1_scale": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add:0",
    "level_1_loc": "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0",
    "target_latent": "agent_0/step_1/reset_core_1_1/MultiLevelSamplerWithARPrior/add_2:0",
}

MODEL_FEATURES = [
    "reward",
    "walker_body_sites",
    "qfrc",
    "qpos",
    "qvel",
    "qacc",
    "xpos",
]

MLP_NETWORK_FEATURES = [
    "level_1_scale",
    "level_1_loc",
    "latent_sample",
    "action_mean",
    # "jacobian_latent_mean",
]

LSTM_NETWORK_FEATURES = [
    "level_1_scale",
    "level_1_loc",
    "latent_sample",
    "lstm_policy_hidden_level_1",
    "lstm_policy_cell_level_1",
    "lstm_policy_hidden_level_2",
    "lstm_policy_cell_level_2",
    "action_mean",
]


FPS = 50
IMAGE_SIZE = [480, 640, 3]
# IMAGE_SIZE = [368, 368, 3]


def load_params(param_path: Text) -> Dict:
    """Load stac parameters for the animal.


    Args:
        param_path (Text): Path to .yaml file specifying animal parameters.

    Returns:
        Dict: Dictionary of stac parameters.
    """
    with open(param_path, "r") as infile:
        try:
            params = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def walker_fn(torque_actuators=False, **kwargs) -> rodent.Rat:
    """Specify the rodent walker.

    Args:
        **kwargs: kwargs for rodent.Rat

    Returns:
        rodent.Rat: Rat walker.
    """
    return rodent.Rat(torque_actuators=torque_actuators, foot_mods=True, **kwargs)


class Experiment:
    def __init__(
        self,
        system: System,
        observer: Observer,
        loop: Loop,
    ):
        self.system = system
        self.observer = observer
        self.loop = loop

    def run(self):
        """Run the environment using comic model."""
        with tf.Session() as sess:
            self.system.load_model(sess)
            graph = tf.get_default_graph()
            timestep, feed_dict, action_output = self.loop.initialize(sess)
            self.loop.loop(sess, action_output, timestep, feed_dict, self.observer)


def parse():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace of command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ref-path",
        dest="ref_path",
        default="./JDM31_rat7m/data/total.hdf5",
        help="Path to dataset containing reference trajectories.",
    )
    parser.add_argument(
        "--import-dir",
        dest="model_dir",
        default="rodent_tracking_model_16212280_3_no_noise",
        help="Path to model import directory.",
    )
    parser.add_argument(
        "--stac-params",
        dest="stac_params",
        help="Path to stac params (.yaml).",
    )
    parser.add_argument(
        "--offset-path",
        dest="offset_path",
        help="Path to stac output with offset(.p).",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        default=".",
        help="Path to save directory.",
    )
    parser.add_argument(
        "--ref-steps",
        dest="ref_steps",
        type=ast.literal_eval,
        default=(1, 2, 3, 4, 5),
        help="Number of steps to look ahead in the reference trajectory. ",
    )
    parser.add_argument(
        "--termination-error-threshold",
        dest="termination_error_threshold",
        type=float,
        default=0.25,
        help="Termination error threshold.",
    )
    parser.add_argument(
        "--min-steps",
        dest="min_steps",
        type=int,
        default=10,
        help="Minimum number of steps to take in environment.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default="jdm31_rat7m",
        help="Name of trajectory dataset to use. Must be registered in locomotion/tasks/reference_pose/datasets.py.",
    )
    parser.add_argument(
        "--reward-type",
        dest="reward_type",
        default="rat_mimic_force",
        help="Reward function name. See locomotion/tasks/reference_pose/rewards.py",
    )
    parser.add_argument(
        "--physics-timestep",
        dest="physics_timestep",
        type=float,
        default=0.001,
        help="Physics timestep.",
    )
    parser.add_argument(
        "--body-error-multiplier",
        dest="body_error_multiplier",
        type=int,
        default=10,
        help="Body error multiplier.",
    )
    parser.add_argument(
        "--video-length",
        dest="video_length",
        type=int,
        default=2500,
        help="Timesteps to include per video. Also sets checkpoint frequency",
    )
    parser.add_argument(
        "--start-step",
        dest="start_step",
        type=int,
        default=0,
        help="Time step in trajectory to start rollout.",
    )
    parser.add_argument(
        "--end-step",
        dest="end_step",
        type=int,
        default=2500,
        help="Time step in trajectory to finish rollout.",
    )
    return parser.parse_args()


def npmp_embed_single_batch():
    """CLI Entrypoint to embed a single batch in a multi processing system.

    Embeds a single batch in a batch array. Reads batch options from _batch_args.p.

    Optional Args:
        stac_params (Text): Path to stac params (.yaml).
        offset_path (Text): Path to stac output with offset(.p).

    Deleted Parameters:
        ref_path (Text): Path to .hdf5 reference trajectories.
        save_dir (Text): Folder in which to save .mat files.
        dataset (Text): Name of dataset registered in dm_control.
        model_dir (Text): Path to rodent tracking model.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref_path",
        help="Path to .hdf5 reference trajectories.",
    )
    parser.add_argument(
        "save_dir",
        help="Folder in which to save .mat files.",
    )
    parser.add_argument(
        "dataset",
        help="Name of dataset registered in dm_control.",
    )
    parser.add_argument(
        "model_dir",
        help="path to rodent tracking model.",
    )
    parser.add_argument(
        "--stac-params",
        dest="stac_params",
        help="Path to stac params (.yaml).",
    )
    parser.add_argument(
        "--offset-path",
        dest="offset_path",
        help="Path to stac output with offset(.p).",
    )
    parser.add_argument(
        "--batch-file",
        dest="batch_file",
        default="_batch_args.p",
        help="Path to stac output with offset(.p).",
    )
    parser.add_argument(
        "--use-open-loop",
        dest="use_open_loop",
        default=False,
        help="If True, use open loop.",
    )
    args = parser.parse_args()

    # Load in parameters to modify
    with open(args.batch_file, "rb") as file:
        batch_args = pickle.load(file)
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    batch_args = batch_args[task_id]
    print(batch_args)
    # args = args.__dict__
    # del args["batch_file"]

    system = System(
        ref_path=args.ref_path,
        model_dir=args.model_dir,
        dataset=args.dataset,
        stac_params=args.stac_params,
        offset_path=args.offset_path,
        start_step=batch_args["start_step"],
        torque_actuators=batch_args["torque_actuators"],
    )
    if batch_args["lstm"]:
        observer = LstmObserver(system.environment, args.save_dir)
        feeder = LstmFeeder()
    else:
        observer = MlpObserver(system.environment, args.save_dir)
        feeder = MlpFeeder()

    if args.use_open_loop:
        loop = OpenLoop(
            system.environment, feeder, batch_args["start_step"], args.video_length
        )
    else:
        loop = ClosedLoop(
            system.environment, feeder, batch_args["start_step"], args.video_length
        )

    exp = Experiment(system, observer, loop)
    exp.run()


# def npmp_embed():
#     """Embed entire clip."""
#     args = parse()
#     npmp = NpmpEmbedder(**args.__dict__)
#     npmp.embed()
