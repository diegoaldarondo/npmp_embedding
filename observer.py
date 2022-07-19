"""Specify observers that handle data recording"""
from dm_control import composer
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
import numpy as np
import imageio
from scipy.io import savemat
import os
from typing import Dict, List, Text, Tuple

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


class Observer:
    """Class to ovserve and record data."""

    def __init__(
        self,
        env: composer.Environment,
        save_dir: Text,
        network_features: List,
        network_observations: List = OBSERVATIONS,
        model_features: List = MODEL_FEATURES,
        seg_frames: bool = False,
        camera_id: Text = "Camera1",
    ):
        """Intialize observer

        Args:
            env (composer.Environment): System environment
            save_dir (Text): Save directory
            network_features (List): List of network features to record
            network_observations (List): List of observations the network makes that you wish to record
            model_features (List): List of model (body) features you wish to record
            observables (List): List of data types to observe
            seg_frames (bool, optional): If true, segment background in images.
                Defaults to False.
            camera_id (Text, optional): Camera name. Defaults to "Camera1".
        """
        self.env = env
        self.save_dir = save_dir
        self.data = {}
        self.cam_list = []
        self.seg_frames = seg_frames
        self.camera_id = camera_id
        self.scene_option = None
        self.network_features = network_features
        self.network_observations = network_observations
        self.model_features = model_features
        self.setup_data_dicts()
        self.setup_scene_rendering()
        self.env.reset()
        self.setup_network_features()

    def setup_data_dicts(self):
        """Initialize data dictionary"""
        for data_type in self.model_features:
            self.data[data_type] = []
        for obs in self.network_observations:
            self.data[obs] = []
        self.data["reset"] = []

    def setup_network_features(self):
        """Set up data dictionary with network features

        Args:
            network_features (List): List of data types to observe
        """
        for feature in self.network_features:
            self.data[feature] = []

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

        # To get the camera quaternion, you need to convert the matlab camera rotation matrix:
        # In Matlab:
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

    def observe(
        self, action_output_np: Dict, timestep, record_physical_features: bool = True
    ):
        """Observe model features.

        Args:
            action_output_np (Dict): Dict of numpy arrays with network features.
            timestep (TYPE): Timestep of the environment.
            record_physical_features (bool, optional): If true, record physical features.
        """
        for feature in self.network_features:
            self.data[feature].append(action_output_np[feature].copy())

        # Record the reward
        if timestep.reward is None:
            self.data["reward"].append(0.0)
        else:
            self.data["reward"].append(timestep.reward)

        # Record model features.
        if record_physical_features:
            self.data["walker_body_sites"].append(
                np.copy(self.env.physics.bind(self.env.task._walker.body_sites).xpos[:])
            )
            self.data["qfrc"].append(
                np.copy(self.env.physics.named.data.qfrc_actuator[:])
            )
            self.data["qpos"].append(np.copy(self.env.physics.named.data.qpos[:]))
            self.data["qvel"].append(np.copy(self.env.physics.named.data.qvel[:]))
            self.data["qacc"].append(np.copy(self.env.physics.named.data.qacc[:]))
            self.data["xpos"].append(np.copy(self.env.physics.named.data.xpos[:]))

        for obs in self.network_observations:
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
        for k, v in self.data.items():
            self.data[k] = np.array(v)
        self.data["action_names"] = action_names
        savemat(save_path, self.data)


class MlpObserver(Observer):
    """Mlp network observer"""

    def __init__(self, environment: composer.Environment, save_dir: Text):
        """Initialize Mlp network observer

        Args:
            environment (composer.Environment): System environment
            save_dir (Text): Save directory
        """
        super().__init__(environment, save_dir, MLP_NETWORK_FEATURES)


class LstmObserver(Observer):
    """Lstm network observer"""

    def __init__(self, environment: composer.Environment, save_dir: Text):
        """Initialize Lstm network observer

        Args:
            environment (composer.Environment): System environment
            save_dir (Text): Save directory
        """
        super().__init__(environment, save_dir, LSTM_NETWORK_FEATURES)


class NullObserver:
    def __init__(self):
        method_list = [
            method for method in dir(Observer) if method.startswith("__") is False
        ]
        for method in method_list:
            setattr(self, method, self.none)

    def none():
        pass
