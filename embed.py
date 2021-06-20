"""Embed out of sample data using npmp architecture.

Attributes:
    DATA_TYPES (List): Data saved for all models
    FPS (int): Frame rate of the video
    IMAGE_SIZE (List): Dimensions of images in video
    LSTM_ACTIONS (Dict): Lstm actions and node names
    LSTM_DATA_TYPES (Dict): Lstm data to save
    LSTM_FULL_INPUTS (Dict): Lstm inputs and node names
    LSTM_STATES (List): Lstm state features
    mjlib (module): Shorthand for mjbindings.mjlib
    MLP_ACTIONS (Dict): Mlp actions and node names
    MLP_DATA_TYPES (Dict): Mlp data to save
    MLP_FULL_INPUTS (Dict): MLP inputs and node names
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

MLP_FULL_INPUTS = {
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

LSTM_FULL_INPUTS = {
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

DATA_TYPES = [
    "reward",
    "walker_body_sites",
    "qfrc",
    "qpos",
    "qvel",
    "qacc",
    "xpos",
]

MLP_DATA_TYPES = [
    "level_1_scale",
    "level_1_loc",
    "latent_sample",
    "action_mean",
    # "jacobian_latent_mean",
]

LSTM_DATA_TYPES = [
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


class Observer:
    def __init__(
        self,
        env,
        save_dir,
        seg_frames: bool = False,
        camera_id: Text = "Camera1",
    ):
        self.env = env
        self.save_dir = save_dir
        self.data = {}
        self.cam_list = []
        self.seg_frames = seg_frames
        self.camera_id = camera_id
        self.scene_option = None
        self.setup_data_dicts()
        self.setup_scene_rendering()
        self.env.reset()

    def setup_data_dicts(self):
        """Initialize data dictionary"""
        for data_type in DATA_TYPES:
            self.data[data_type] = []
        for obs in OBSERVATIONS:
            self.data[obs] = []
        self.data["reset"] = []

    def setup_model_observables(self, observables):
        self.data_types = observables
        for data_type in self.data_types:
            self.data[data_type] = []

    def setup_scene_rendering(self):
        """Set the scene to segment the frames or not."""
        # Add the camera
        # self.environment.task._arena._mjcf_root.worldbody.add(
        #     "camera",
        #     name="CameraE",
        #     pos=[-0.0452, 1.5151, 0.3174],
        #     fovy=50,
        #     quat="0.0010 -0.0202 -0.7422 -0.6699",
        # )

        # To get the quaternion, you need to convert the matlab camera rotation matrrix:
        # eul = rotm2eul(r,'ZYX')
        # eul(3) = eul(3) + pi
        # quat = rotm2quat(eul,'ZYX')
        # This converts matlab definition of the camera frame (X right Y down Z forward)
        # to the mujoco version (X right, Y up, Z backward)
        self.env.task._arena._mjcf_root.worldbody.add(
            "camera",
            name="Camera1",
            pos=[-0.8781364, 0.3775911, 0.4291190],
            fovy=50,
            quat="0.5353    0.3435   -0.4623   -0.6178",
        )
        if self.seg_frames:
            self.scene_option = wrapper.MjvOption()
            self.env.physics.model.skin_rgba[0][3] = 0.0
            self.scene_option.geomgroup[2] = 0
            self.scene_option._ptr.contents.flags[
                enums.mjtVisFlag.mjVIS_TRANSPARENT
            ] = False
            self.scene_option._ptr.contents.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
            self.scene_option._ptr.contents.flags[
                enums.mjtVisFlag.mjVIS_TEXTURE
            ] = False
        else:
            self.scene_option = wrapper.MjvOption()
            self.scene_option.geomgroup[2] = 0
            self.scene_option._ptr.contents.flags[
                enums.mjtVisFlag.mjVIS_TRANSPARENT
            ] = True

    def observe(self, action_output_np: Dict, timestep):
        """Observe model features.

        Args:
            action_output_np (Dict): Description
            timestep (TYPE): Description
        """
        for data_type in self.data_types:
            self.data[data_type].append(action_output_np[data_type])
        self.data["reward"].append(timestep.reward)
        self.data["walker_body_sites"].append(
            np.copy(self.env.physics.bind(self.env.task._walker.body_sites).xpos[:])
        )
        self.data["qfrc"].append(np.copy(self.env.physics.named.data.qfrc_actuator[:]))
        self.data["qpos"].append(np.copy(self.env.physics.named.data.qpos[:]))
        self.data["qvel"].append(np.copy(self.env.physics.named.data.qvel[:]))
        self.data["qacc"].append(np.copy(self.env.physics.named.data.qacc[:]))
        self.data["xpos"].append(np.copy(self.env.physics.named.data.xpos[:]))
        for obs in OBSERVATIONS:
            self.data[obs].append(timestep.observation[obs])
        self.data["reset"].append(timestep.last())

    def grab_frame(self):
        """Render a frame normally or with segmentation."""
        if self.seg_frames:
            rgbArr = self.env.physics.render(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                camera_id=self.camera_id,
                scene_option=self.scene_option,
            )
            seg = self.env.physics.render(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                camera_id=self.camera_id,
                scene_option=self.scene_option,
                segmentation=True,
            )
            bkgrd = (seg[:, :, 0] == -1) & (seg[:, :, 1] == -1)
            floor = (seg[:, :, 0] == 0) & (seg[:, :, 1] == 5)
            rgbArr[:, :, 0] *= ~bkgrd[:, :]
            rgbArr[:, :, 1] *= ~bkgrd[:, :]
            rgbArr[:, :, 2] *= ~bkgrd[:, :]
            rgbArr[:, :, 0] *= ~floor[:, :]
            rgbArr[:, :, 1] *= ~floor[:, :]
            rgbArr[:, :, 2] *= ~floor[:, :]
            self.cam_list.append(rgbArr)
        else:
            self.cam_list.append(
                self.env.physics.render(
                    IMAGE_SIZE[0],
                    IMAGE_SIZE[1],
                    camera_id=self.camera_id,
                    # camera_id="walker/close_profile",
                    scene_option=self.scene_option,
                )
            )

    def checkpoint(self, file_name: Text):
        """Checkpoint the observations."""
        self.checkpoint_video(os.path.join(self.save_dir, "video", file_name + ".mp4"))
        self.checkpoint_data(os.path.join(self.save_dir, "logs", file_name + ".mat"))

        # Clear the cam list to keep memory low.
        self.cam_list = []

    def checkpoint_video(self, filename: Text, fps: int = FPS):
        """Write a video to disk.

        Args:
            filename (Text): Filename of video to save.
            fps (int, optional): Frames per second of encoded video.
        """
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        print("Writing to %s" % (filename))
        with imageio.get_writer(filename, fps=fps) as video:
            print(len(self.cam_list))
            for frame in self.cam_list:
                video.append_data(frame)

    def checkpoint_data(self, save_path: Text):
        """Save data checkoint.

        Args:
            save_path (Text): Directory in which to save output.
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        action_names = self.env.physics.named.data.act.axes.row.names
        out_dict = {k: np.array(v) for k, v in self.data.items()}
        out_dict["action_names"] = action_names
        savemat(save_path, out_dict)


class LstmObserver(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_model_observables(LSTM_DATA_TYPES)


class MlpObserver(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_model_observables(MLP_DATA_TYPES)


class NullObserver:
    def __init__(self):
        method_list = [
            method for method in dir(Observer) if method.startswith("__") is False
        ]
        for method in method_list:
            setattr(self, method, self.none)

    def none():
        pass


class Feeder(abc.ABC):
    def __init__(self):
        self.full_inputs = None

    def feed(self, timestep, action_output_np: np.ndarray = None):
        feed_dict = {}
        for obs in OBSERVATIONS:
            feed_dict[self.full_inputs[obs]] = timestep.observation[obs]
        for state in self.states:
            if action_output_np is None:
                feed_dict[self.full_inputs[state]] = np.zeros(
                    self.full_inputs[state].shape
                )
            else:
                feed_dict[self.full_inputs[state]] = action_output_np[state].flatten()
        feed_dict[self.full_inputs["step_type"]] = timestep.step_type
        feed_dict[self.full_inputs["reward"]] = timestep.reward
        feed_dict[self.full_inputs["discount"]] = timestep.discount
        return feed_dict

    @abc.abstractmethod
    def get_inputs(self) -> Dict:
        """Setup full_inputs for the model.

        Returns:
            Dict: full input dict
        """
        pass

    @abc.abstractmethod
    def get_outputs(self) -> Dict:
        """Setup full_inputs for the model.

        Returns:
            Dict: Action output dict
        """
        pass


class LstmFeeder(Feeder):
    def __init__(self, *args, **kwargs):
        self.states = LSTM_STATES
        super().__init__(*args, **kwargs)

    def get_inputs(self, sess: tf.Session) -> Dict:
        """Setup full_inputs for the model.

        Args:
            sess (tf.Session): Current tf session.

        Returns:
            Dict: full input dict
        """
        self.full_inputs = {
            k: sess.graph.get_tensor_by_name(v) for k, v in LSTM_FULL_INPUTS.items()
        }
        return self.full_inputs

    def get_outputs(self, sess: tf.Session) -> Dict:
        """Setup action output for the model.

        Args:
            sess (tf.Session): Current tf session.

        Returns:
            Dict: Action output dict
        """
        action_output = {
            k: sess.graph.get_tensor_by_name(v) for k, v in LSTM_ACTIONS.items()
        }
        return action_output


class MlpFeeder(Feeder):
    def __init__(self, *args, **kwargs):
        self.states = MLP_STATES
        super().__init__(*args, **kwargs)

    def get_inputs(self, sess: tf.Session) -> Dict:
        """Setup full_inputs for the model.

        Args:
            sess (tf.Session): Current tf session.

        Returns:
            Dict: full input dict
        """
        self.full_inputs = {
            k: sess.graph.get_tensor_by_name(v) for k, v in MLP_FULL_INPUTS.items()
        }
        return self.full_inputs

    def get_outputs(self, sess: tf.Session) -> Dict:
        """Setup action output for the model.

        Args:
            sess (tf.Session): Current tf session.

        Returns:
            Dict: Action output dict
        """
        action_output = {
            k: sess.graph.get_tensor_by_name(v) for k, v in MLP_ACTIONS.items()
        }
        action_output["jacobian_latent_mean"] = jacobian(
            sess.graph.get_tensor_by_name(
                "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0"
            ),
            sess.graph.get_tensor_by_name(
                "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0"
            ),
        )
        return action_output


class Loop(abc.ABC):
    def __init__(self, env, feeder: Feeder, start_step: int, video_length: int):
        self.env = env
        self.feeder = feeder
        self.start_step = start_step
        self.video_length = video_length
        self.full_inputs = None
        self.states = None
        self.lstm = False

    def reset(self):
        """Restart an environment from the start_step

        Returns:
            (TYPE): Current timestep
            Dict: Input dictionary with updated values
        """
        timestep = self.env.reset()
        feed_dict = self.feeder.feed(timestep, action_output_np=None)
        return timestep, feed_dict

    def step(self, action_output_np: Dict):
        """Perform a single step within an environment.

        Args:
            action_output_np (Dict): Description

        Returns:
            (TYPE): Current timestep
            Dict: Input dictionary with updated values
        """
        # timestep = self.environment.step(action_output_np["action"])
        timestep = self.env.step(action_output_np["action_mean"])
        feed_dict = self.feeder.feed(timestep, action_output_np)
        return timestep, feed_dict

    @abc.abstractmethod
    def loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
        observer: Observer = NullObserver(),
    ):
        pass

    def initialize(self, sess: tf.Session) -> Tuple:
        """Initialize the loop.

        Args:
            sess (tf.Session): Current tf session

        Returns:
            Tuple: Timestep, feed_dict, action_output
        """
        _ = self.feeder.get_inputs(sess)
        action_output = self.feeder.get_outputs(sess)
        timestep, feed_dict = self.reset()
        return timestep, feed_dict, action_output


class ClosedLoop(Loop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
        observer: Observer = NullObserver(),
    ):
        """Roll-out the model in closed loop with the environment.

        Args:
            sess (tf.Session): Tensorflow session
            action_output (Dict): Dictionary of logged outputs
            timestep (TYPE): Timestep object for the current roll out.
            feed_dict (Dict): Dictionary of inputs
        """
        try:
            for n_step in range(self.video_length):
                print(n_step, flush=True)
                observer.grab_frame()

                # If the task failed, restart at the new step
                if timestep.last():
                    self.env.task.start_step = n_step
                    timestep, feed_dict = self.reset()

                # Get the action and step in the environment
                action_output_np = sess.run(action_output, feed_dict)
                timestep, feed_dict = self.step(action_output_np)

                # Make observations
                observer.observe(action_output_np, timestep)

                # Save a checkpoint of the data and video
                if n_step + 1 == self.video_length:
                    observer.checkpoint(str(self.start_step))
        except IndexError:
            while len(observer.data["reward"]) < self.video_length:
                for data_type in observer.data.keys():
                    observer.data[data_type].append(observer.data[data_type][-1])
            # while len(cam_list) < self.video_length:
            #     self.cam_list.append(self.cam_list[-1])
            observer.checkpoint(str(self.start_step))


class OpenLoop(Loop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
        observer: Observer = NullObserver(),
    ):
        """Roll out the model in open loop with the environment.

        Args:
            sess (tf.Session): Tensorflow session
            action_output (Dict): Dictionary of logged outputs
            timestep (TYPE): Timestep object for the current roll out.
            feed_dict (Dict): Dictionary of inputs
        """
        try:
            for n_step in range(self.video_length):
                observer.grab_frame()

                print(n_step, flush=True)
                self.env.task.start_step = n_step
                timestep, feed_dict = self.reset()

                # Get the action and step in the environment
                action_output_np = sess.run(action_output, feed_dict)
                timestep, feed_dict = self.step(action_output_np)

                # Make observations
                observer.observe(action_output_np, timestep)

                # Save a checkpoint of the data and video
                if n_step + 1 == self.video_length:
                    observer.checkpoint(str(self.start_step))
        except IndexError:
            while len(observer.data["reward"]) < self.video_length:
                for data_type in observer.data.keys():
                    observer.data[data_type].append(observer.data[data_type][-1])
            # while len(cam_list) < self.video_length:
            #     self.cam_list.append(self.cam_list[-1])
            observer.checkpoint(str(self.start_step))


class System:
    """A system includes an environment with tasks and walkers."""

    def __init__(
        self,
        ref_path: Text,
        model_dir: Text,
        dataset: Text,
        stac_params: Text,
        offset_path: Text = None,
        arena: composer.Arena = floors.Floor(size=(10.0, 10.0)),
        ref_steps: Tuple = (1, 2, 3, 4, 5),
        termination_error_threshold: float = 0.25,
        min_steps: int = 10,
        reward_type: Text = "rat_mimic_force",
        physics_timestep: float = 0.001,
        body_error_multiplier: float = 10,
        video_length: int = 2500,
        min_action: float = -1.0,
        max_action: float = 1.0,
        start_step: int = 0,
        torque_actuators: bool = False,
    ):
        """Utility class to roll out model for new snippets.

        Attributes:
            arena (composer.Arena, optional): Arena in which to perform roll out.
            body_error_multiplier (float): Scaling factor for body error.
            cam_list (list): List of rendered video frames over time.
            camera_id (Text): Name of the camera to use for rendering.
            data (Dict): Observed data
            dataset (Text): Name of dataset registered in dm_control.
            end_step (int): Last step of video
            environment (composer.Environment): Environment for roll out.
            full_inputs (Dict): All inputs to model.
            model_dir (Text): Directory of trained model.
            lstm (bool): Set to true if rolling out an LSTM model.
            max_action (float): Maximum value of action.
            min_action (float): Minimum value of action.
            min_steps (int): Minimum number of steps in a roll out.
            offset_path (Text): Path to offsets .pickle file.
            physics_timestep (float): Timestep for physics calculations
            ref_path (Text): Path to reference snippet.
            ref_steps (Tuple): Reference steps. e.g (1, 2, 3, 4, 5)
            reward_type (Text): Type of reward. Default "rat_mimic_force"
            save_dir (Text): Path to saving directory.
            scene_option (wrapper.MjvOption): MjvOptions for scene rendering.
            seg_frames (bool): If True, segment animal from background in video.
            stac_params (Text): Path to stack params.yaml file.
            start_step (int): First step of video
            termination_error_threshold (float, optional): Error threshold at which to stop roll out.
            use_open_loop (bool): If True, use open-loop during roll out.
            video_length (int): Length of snippet in frames.
        """

        self.ref_path = ref_path
        self.model_dir = model_dir
        self.arena = arena
        self.stac_params = stac_params
        self.offset_path = offset_path
        self.ref_steps = ref_steps
        self.termination_error_threshold = termination_error_threshold
        self.min_steps = min_steps
        self.dataset = dataset
        self.reward_type = reward_type
        self.physics_timestep = physics_timestep
        self.body_error_multiplier = body_error_multiplier
        self.video_length = video_length
        self.min_action = min_action
        self.max_action = max_action
        self.start_step = start_step
        self.torque_actuators = torque_actuators

        # Set up the stac parameters to compute the inferred keypoints
        # in CoMic rollouts.
        params = load_params(self.stac_params)
        self.setup_environment(params)
        self.setup_offsets(params)

    def setup_environment(self, params: Dict):
        """Setup task and environment

        Args:
            params (Dict): Stac parameters dict
        """
        task = tracking.SingleClipTracking(
            clip_id="clip_%d" % (self.start_step),
            clip_length=self.video_length + np.max(self.ref_steps) + 1,
            walker=lambda **kwargs: walker_fn(
                params=params, torque_actuators=self.torque_actuators, **kwargs
            ),
            arena=self.arena,
            ref_path=self.ref_path,
            ref_steps=self.ref_steps,
            termination_error_threshold=self.termination_error_threshold,
            dataset=self.dataset,
            min_steps=self.min_steps,
            reward_type=self.reward_type,
            physics_timestep=self.physics_timestep,
            body_error_multiplier=self.body_error_multiplier,
        )
        self.environment = action_scale.Wrapper(
            composer.Environment(task), self.min_action, self.max_action
        )

    def setup_offsets(self, params: Dict):
        """Set the keypoint offsets.

        Args:
            params (Dict): Stac parameters dict.
        """
        if self.offset_path is not None and self.stac_params is not None:
            params["offset_path"] = self.offset_path
        with open(params["offset_path"], "rb") as f:
            in_dict = pickle.load(f)
        sites = self.environment.task._walker.body_sites
        self.environment.physics.bind(sites).pos[:] = in_dict["offsets"]
        for n_site, p in enumerate(self.environment.physics.bind(sites).pos):
            sites[n_site].pos = p

    def load_model(self, sess: tf.Session):
        tf.saved_model.loader.load(sess, ["tag"], self.model_dir)


class Experiment:
    def __init__(
        self,
        system: System,
        observer: Observer,
        feeder: Feeder,
        loop: Loop,
    ):
        self.system = system
        self.observer = observer
        self.feeder = feeder
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

    exp = Experiment(system, observer, feeder, loop)
    exp.run()

    # npmp = NpmpEmbedder(**args, **batch_args)
    # npmp.embed()


def npmp_embed():
    """Embed entire clip."""
    args = parse()
    npmp = NpmpEmbedder(**args.__dict__)
    npmp.embed()
