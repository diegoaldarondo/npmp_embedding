"""Embed out of sample data using npmp architecture.

Attributes:
    FPS (int): Frame rate of the video
    IMAGE_SIZE (list): Dimensions of images in video
    mjlib (module): Shorthand for mjbindings.mjlib
    OBSERVATIONS (list): Model observations
    STATES (list): Model States
"""
import dm_control
import h5py
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.walkers import rodent
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control import composer
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import transformations
from scipy.io import loadmat
import pickle
import mocap_preprocess
import numpy as np
import matplotlib.pyplot as plt
import imageio
import base64
import IPython
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
import umap
from dm_control.suite.wrappers import action_scale
from scipy.io import savemat
import argparse
import ast
import os
import pickle
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
import yaml
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
from typing import Dict, List, Text, Tuple

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

STATES = [
    "latent",
    "target_latent",
    "dummy_core_state",
    "dummy_target_core_state",
    "dummy_policy_state_level_1",
    "dummy_policy_state_level_2",
    "dummy_target_policy_state_level_1",
    "dummy_target_policy_state_level_2",
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


def walker_fn(**kwargs) -> rodent.Rat:
    """Specify the rodent walker.

    Args:
        **kwargs: kwargs for rodent.Rat

    Returns:
        rodent.Rat: Rat walker.
    """
    return rodent.Rat(torque_actuators=False, foot_mods=True, **kwargs)


class NpmpEmbedder:
    """Utility class to roll out model for new snippets.

    Attributes:
        action_mean (list): List of actions over time.
        arena (composer.Arena, optional): Arena in which to perform roll out.
        body_error_multiplier (float): Scaling factor for body error.
        cam_list (list): List of rendered video frames over time.
        camera_id (Text): Name of the camera to use for rendering.
        dataset (Text): Name of dataset registered in dm_control.
        end_step (int): Last step of video
        environment (composer.Environment): Environment for roll out.
        full_inputs (Dict): All inputs to model.
        import_dir (Text): Directory of trained model.
        jacobian_latent_mean (list): List of Jacobian of action mean w/r latent mean over time.
        jacobian_latent_scale (list): List of Jacobian of action mean w/r latent scale over time.
        latent_list (list): List of latent samples over time (samce as latent_sample)
        latent_mean_list (list): List of latent mean over time (same as level 1 loc)
        latent_noise (list): List of latent noise over time.
        latent_sample (list): List of latent samples over time.
        level_1_loc (list): List of level 1 locs over time.
        level_1_scale (list): List of level 1 scales over time.
        max_action (float): Maximum value of action.
        min_action (float): Minimum value of action.
        min_steps (int): Minimum number of steps in a roll out.
        offset_path (Text): Path to offsets .pickle file.
        physics_timestep (float): Timestep for physics calculations
        prior_mean (list): List of prior means over time.
        ref_path (Text): Path to reference snippet.
        ref_steps (Tuple): Reference steps. e.g (1, 2, 3, 4, 5)
        rew_list (list): List of rewards over time.
        reward_type (Text): Type of reward. Default "rat_mimic_force"
        save_dir (Text): Path to saving directory.
        scene_option (wrapper.MjvOption): MjvOptions for scene rendering.
        seg_frames (bool): If True, segment animal from background in video.
        stac_params (Text): Path to stack params.yaml file.
        start_step (int): First step of video
        termination_error_threshold (float, optional): Error threshold at which to stop roll out.
        use_open_loop (bool): If True, use open-loop during roll out.
        video_length (int): Length of snippet in frames.
        walker_body_sites (list): List of keypoint sites on the body.
    """

    def __init__(
        self,
        ref_path: Text,
        save_dir: Text,
        import_dir: Text,
        dataset: Text,
        stac_params: Text = None,
        offset_path: Text = None,
        arena: composer.Arena = floors.Floor(size=(10.0, 10.0)),
        ref_steps: Tuple = (1, 2, 3, 4, 5),
        termination_error_threshold: float = 0.25,
        min_steps: int = 10,
        reward_type: Text = "rat_mimic_force",
        physics_timestep: float = 0.001,
        body_error_multiplier: float = 10,
        video_length: int = 2500,
        end_step: int = 210000,
        min_action: float = -1.0,
        max_action: float = 1.0,
        start_step: int = 0,
        seg_frames: bool = False,
        camera_id: Text = "walker/close_profile",
        use_open_loop: bool = False,
    ):
        """Initialize the embedder.

        Args:
            ref_path (Text): Path to reference snippet.
            save_dir (Text): Path to saving directory.
            import_dir (Text): Directory of trained model.
            dataset (Text): Name of dataset registered in dm_control.
            stac_params (Text, optional): Path to stack params.yaml file.
            offset_path (Text, optional): Path to offsets .pickle file.
            arena (composer.Arena, optional): Arena in which to perform roll out.
            ref_steps (Tuple, optional): Reference steps. e.g (1, 2, 3, 4, 5)
            termination_error_threshold (float, optional): Error threshold at which to stop roll out.
            min_steps (int, optional): Minimum number of steps in a roll out.
            reward_type (Text, optional): Type of reward. Default "rat_mimic_force"
            physics_timestep (float, optional): Timestep for physics calculations
            body_error_multiplier (float, optional): Scaling factor for body error.
            video_length (int, optional): Length of snippet in frames.
            end_step (int, optional): Last step of video
            min_action (float, optional): Minimum value of action.
            max_action (float, optional): Maximum value of action.
            start_step (int, optional): First step of video
            seg_frames (bool, optional): If True, segment animal from background in video.
            camera_id (Text, optional): Name of the camera to use for rendering.
            use_open_loop (bool, optional): If True, use open-loop during roll out.
        """
        self.latent_list = None
        self.latent_mean_list = None
        self.rew_list = None
        self.cam_list = None
        self.full_inputs = None
        self.ref_path = ref_path
        self.import_dir = import_dir
        self.save_dir = save_dir
        self.arena = arena
        self.stac_params = stac_params
        self.offset_path = offset_path
        # arena._ground_geom.pos = [0.0, 0.0, -0.1]
        self.ref_steps = ref_steps
        self.termination_error_threshold = termination_error_threshold
        self.min_steps = min_steps
        self.dataset = dataset
        self.reward_type = reward_type
        self.physics_timestep = physics_timestep
        self.body_error_multiplier = body_error_multiplier
        self.video_length = video_length
        self.end_step = end_step
        self.min_action = min_action
        self.max_action = max_action
        self.start_step = start_step
        self.seg_frames = seg_frames
        self.camera_id = camera_id
        self.use_open_loop = use_open_loop

        # task = tracking.LongClipTracking(
        #     self.start_step,
        #     walker=walker_fn,
        #     arena=self.arena,
        #     ref_path=self.ref_path,
        #     ref_steps=self.ref_steps,
        #     termination_error_threshold=self.termination_error_threshold,
        #     dataset=self.dataset,
        #     min_steps=self.min_steps,
        #     reward_type=self.reward_type,
        #     physics_timestep=self.physics_timestep,
        #     body_error_multiplier=self.body_error_multiplier,
        # )

        # Set up the stac parameters to compute the inferred mocap reference
        # points in CoMic rollouts.
        if self.stac_params is not None:
            params = load_params(self.stac_params)

        task = tracking.SingleClipTracking(
            clip_id="clip_%d" % (self.start_step),
            clip_length=self.video_length + np.max(self.ref_steps) + 1,
            walker=lambda **kwargs: walker_fn(params=params, **kwargs),
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
        print(task.clip_id, flush=True)
        self.environment = action_scale.Wrapper(
            composer.Environment(task), self.min_action, self.max_action
        )
        # self.environment.task.start_step = self.start_step
        self.environment.task._arena._mjcf_root.worldbody.add(
            "camera",
            name="CameraE",
            pos=[-0.0452, 1.5151, 0.3174],
            fovy=50,
            quat="0.0010 -0.0202 -0.7422 -0.6699",
        )

        if self.seg_frames:
            self.scene_option = wrapper.MjvOption()
            self.environment.physics.model.skin_rgba[0][3] = 0.0
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
        self.environment.reset()

        # Get the offsets.
        if self.offset_path is not None and self.stac_params is not None:
            params["offset_path"] = self.offset_path
        with open(params["offset_path"], "rb") as f:
            in_dict = pickle.load(f)
        sites = self.environment.task._walker.body_sites
        self.environment.physics.bind(sites).pos[:] = in_dict["offsets"]
        for n_site, p in enumerate(self.environment.physics.bind(sites).pos):
            sites[n_site].pos = p

    def setup_keypoint_sites(self):
        """Add keypoint sites to estimate original keypoint positions."""
        # Set up the stac parameters to compute the inferred mocap reference
        # points in CoMic rollouts.
        if self.stac_params is not None:
            params = load_params(self.stac_params)

        if self.offset_path is not None and self.stac_params is not None:
            params["offset_path"] = self.offset_path
        # Add keypoint sites to the mjcf model, and a reference to the sites as
        # an attribute for easier access. This allows us to save the reference
        # markers during CoMic roll out.
        self.environment.task._walker.body_sites = []
        for key, v in params["_KEYPOINT_MODEL_PAIRS"].items():
            parent = self.environment.task._walker._mjcf_root.find("body", v)
            pos = params["_KEYPOINT_INITIAL_OFFSETS"][key]
            site = parent.add(
                "site",
                name=key,
                type="sphere",
                size=[0.005],
                rgba="0 0 0 1",
                pos=pos,
                group=3,
            )
            self.environment.task._walker.body_sites.append(site)

        # Get the offsets.
        with open(params["offset_path"], "rb") as f:
            in_dict = pickle.load(f)

        sites = self.environment.task._walker.body_sites

        self.environment.physics.bind(sites).pos[:] = in_dict["offsets"]
        for n_site, p in enumerate(self.environment.physics.bind(sites).pos):
            sites[n_site].pos = p

    def checkpoint_eval_video(self, filename: Text, fps: int = FPS):
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

    def reset_rollout(self):
        """Restart an environment from the start_step

        Returns:
            (TYPE): Current timestep
            Dict: Input dictionary with updated values
        """
        timestep = self.environment.reset()
        feed_dict = {}
        for obs in OBSERVATIONS:
            feed_dict[self.full_inputs[obs]] = timestep.observation[obs]
        for state in STATES:
            feed_dict[self.full_inputs[state]] = np.zeros(self.full_inputs[state].shape)
        feed_dict[self.full_inputs["step_type"]] = timestep.step_type
        feed_dict[self.full_inputs["reward"]] = timestep.reward
        feed_dict[self.full_inputs["discount"]] = timestep.discount
        return timestep, feed_dict

    def step_rollout(self, action_output_np: Dict):
        """Perform a single step within an environment.

        Args:
            action_output_np (Dict): Description

        Returns:
            (TYPE): Current timestep
            Dict: Input dictionary with updated values
        """
        # timestep = self.environment.step(action_output_np["action"])
        timestep = self.environment.step(action_output_np["action_mean"])
        feed_dict = {}
        for obs in OBSERVATIONS:
            feed_dict[self.full_inputs[obs]] = timestep.observation[obs]
        for state in STATES:
            feed_dict[self.full_inputs[state]] = action_output_np[state].flatten()
        feed_dict[self.full_inputs["step_type"]] = timestep.step_type
        feed_dict[self.full_inputs["reward"]] = timestep.reward
        feed_dict[self.full_inputs["discount"]] = timestep.discount
        return timestep, feed_dict

    def grab_frame(self):
        """Render a frame normally or with segmentation."""
        if self.seg_frames:
            rgbArr = self.environment.physics.render(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                camera_id=self.camera_id,
                scene_option=self.scene_option,
            )
            seg = self.environment.physics.render(
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
                self.environment.physics.render(
                    IMAGE_SIZE[0],
                    IMAGE_SIZE[1],
                    camera_id=self.camera_id,
                    # camera_id="walker/close_profile",
                    scene_option=self.scene_option,
                )
            )

    def embed(self):
        """Embed trajectories using npmp model."""
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, ["tag"], self.import_dir)
            graph = tf.get_default_graph()

            self.full_inputs = {
                "step_type": sess.graph.get_tensor_by_name("step_type_2:0"),
                "reward": sess.graph.get_tensor_by_name("reward_2:0"),
                "discount": sess.graph.get_tensor_by_name("discount_1:0"),
                "walker/actuator_activation": sess.graph.get_tensor_by_name(
                    "walker/actuator_activation_1:0"
                ),
                "walker/appendages_pos": sess.graph.get_tensor_by_name(
                    "walker/appendages_pos_1:0"
                ),
                "walker/body_height": sess.graph.get_tensor_by_name(
                    "walker/body_height_1:0"
                ),
                "walker/end_effectors_pos": sess.graph.get_tensor_by_name(
                    "walker/end_effectors_pos_1:0"
                ),
                "walker/joints_pos": sess.graph.get_tensor_by_name(
                    "walker/joints_pos_1:0"
                ),
                "walker/joints_vel": sess.graph.get_tensor_by_name(
                    "walker/joints_vel_1:0"
                ),
                "walker/sensors_accelerometer": sess.graph.get_tensor_by_name(
                    "walker/sensors_accelerometer_1:0"
                ),
                "walker/sensors_force": sess.graph.get_tensor_by_name(
                    "walker/sensors_force_1:0"
                ),
                "walker/sensors_gyro": sess.graph.get_tensor_by_name(
                    "walker/sensors_gyro_1:0"
                ),
                "walker/sensors_torque": sess.graph.get_tensor_by_name(
                    "walker/sensors_torque_1:0"
                ),
                "walker/sensors_touch": sess.graph.get_tensor_by_name(
                    "walker/sensors_touch_1:0"
                ),
                "walker/sensors_velocimeter": sess.graph.get_tensor_by_name(
                    "walker/sensors_velocimeter_1:0"
                ),
                "walker/tendons_pos": sess.graph.get_tensor_by_name(
                    "walker/tendons_pos_1:0"
                ),
                "walker/tendons_vel": sess.graph.get_tensor_by_name(
                    "walker/tendons_vel_1:0"
                ),
                "walker/world_zaxis": sess.graph.get_tensor_by_name(
                    "walker/world_zaxis_1:0"
                ),
                "walker/reference_rel_bodies_pos_local": sess.graph.get_tensor_by_name(
                    "walker/reference_rel_bodies_pos_local_1:0"
                ),
                "walker/reference_rel_bodies_quats": sess.graph.get_tensor_by_name(
                    "walker/reference_rel_bodies_quats_1:0"
                ),
                "dummy_core_state": sess.graph.get_tensor_by_name("state_9:0"),
                "dummy_target_core_state": sess.graph.get_tensor_by_name("state_10:0"),
                "dummy_policy_state_level_1": sess.graph.get_tensor_by_name(
                    "state_11:0"
                ),
                "dummy_policy_state_level_2": sess.graph.get_tensor_by_name(
                    "state_12:0"
                ),
                "dummy_target_policy_state_level_1": sess.graph.get_tensor_by_name(
                    "state_14:0"
                ),
                "dummy_target_policy_state_level_2": sess.graph.get_tensor_by_name(
                    "state_15:0"
                ),
                "latent": sess.graph.get_tensor_by_name("state_13:0"),
                "target_latent": sess.graph.get_tensor_by_name("state_16:0"),
            }

            action_output = {
                "action": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag/sample/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_chain_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_scale_matvec_linear_operator/forward/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_actor_head_MultivariateNormalDiag_shift/forward/add:0"
                ),
                "dummy_core_state": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core/Select:0"
                ),
                "dummy_target_core_state": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_2/Select:0"
                ),
                "dummy_policy_state_level_1": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/Select:0"
                ),
                "dummy_policy_state_level_2": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/Select_1:0"
                ),
                "dummy_target_policy_state_level_1": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1_1/Select:0"
                ),
                "dummy_target_policy_state_level_2": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1_1/Select_1:0"
                ),
                "latent": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0"
                ),
                "latent_mean": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0"
                ),
                "target_latent": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1_1/MultiLevelSamplerWithARPrior/add_2:0"
                ),
                "prior_mean": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/mul_1:0"
                ),
                # "latent_noise": sess.graph.get_tensor_by_name(
                #     "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_MultivariateNormalDiag/sample/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_MultivariateNormalDiag_chain_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_MultivariateNormalDiag_shift_of_agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_MultivariateNormalDiag_scale_matvec_linear_operator/forward/agent_0_step_1_reset_core_1_MultiLevelSamplerWithARPrior_MultivariateNormalDiag_shift/forward/add:0"
                # ),
                "level_1_scale": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add:0"
                ),
                "level_1_loc": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0"
                ),
                "latent_sample": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add_2:0"
                ),
                "action_mean": sess.graph.get_tensor_by_name(
                    "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0"
                ),
                "jacobian_latent_mean": jacobian(
                    sess.graph.get_tensor_by_name(
                        "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0"
                    ),
                    sess.graph.get_tensor_by_name(
                        "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:0"
                    ),
                ),
                "jacobian_latent_scale": jacobian(
                    sess.graph.get_tensor_by_name(
                        "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/actor_head/Tanh:0"
                    ),
                    sess.graph.get_tensor_by_name(
                        "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/add:0"
                    ),
                ),
            }

            timestep, feed_dict = self.reset_rollout()
            self.cam_list = []
            self.latent_list = []
            self.prior_mean = []
            self.latent_noise = []
            self.level_1_scale = []
            self.level_1_loc = []
            self.latent_sample = []
            self.latent_mean_list = []
            self.action_mean = []
            self.jacobian_latent_mean = []
            self.jacobian_latent_scale = []
            self.rew_list = []
            self.walker_body_sites = []
            if self.use_open_loop:
                self.open_loop(sess, action_output, timestep, feed_dict)
            else:
                self.closed_loop(sess, action_output, timestep, feed_dict)

            # TODO(Why is this here?)
            action_output_np = sess.run(action_output, feed_dict)

    def closed_loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
    ):
        """Roll-out the model in closed loop with the environment.

        Args:
            sess (tf.Session): Tensorflow session
            action_output (Dict): Dictionary of logged outputs
            timestep (TYPE): Timestep object for the current roll out.
            feed_dict (Dict): Dictionary of inputs
        """
        for n_step in range(self.video_length):
            print(n_step, flush=True)
            self.grab_frame()

            # If the task failed, restart at the new step
            if timestep.last():
                self.environment.task.start_step = n_step
                timestep, feed_dict = self.reset_rollout()

            # Get the action and step in the environment
            action_output_np = sess.run(action_output, feed_dict)
            timestep, feed_dict = self.step_rollout(action_output_np)

            self.walker_body_sites.append(
                np.copy(
                    self.environment.physics.bind(
                        self.environment.task._walker.body_sites
                    ).xpos[:]
                )
            )
            self.rew_list.append(timestep.reward)
            self.latent_list.append(action_output_np["latent"].flatten())
            self.prior_mean.append(action_output_np["prior_mean"].flatten())
            # self.latent_noise.append(action_output_np["latent_noise"].flatten())
            self.level_1_scale.append(action_output_np["level_1_scale"].flatten())
            self.level_1_loc.append(action_output_np["level_1_loc"].flatten())
            self.latent_sample.append(action_output_np["latent_sample"].flatten())
            self.latent_mean_list.append(action_output_np["latent_mean"].flatten())
            self.action_mean.append(action_output_np["action_mean"])
            self.jacobian_latent_mean.append(action_output_np["jacobian_latent_mean"])
            self.jacobian_latent_scale.append(action_output_np["jacobian_latent_scale"])

            # Save a checkpoint of the data and video
            if n_step + 1 == self.video_length:
                self.checkpoint_eval_video(
                    os.path.join(self.save_dir, "video", str(self.start_step) + ".mp4")
                )
                self.save_checkpoint(
                    os.path.join(self.save_dir, "logs", str(self.start_step) + ".mat")
                )

                # Clear the cam list to keep memory low.
                self.cam_list = []

    def open_loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
    ):
        """Roll out the model in open loop with the environment.

        Args:
            sess (tf.Session): Tensorflow session
            action_output (Dict): Dictionary of logged outputs
            timestep (TYPE): Timestep object for the current roll out.
            feed_dict (Dict): Dictionary of inputs
        """
        # count = self.start_step
        for n_step in range(self.video_length):
            self.grab_frame()

            print(n_step, flush=True)
            self.environment.task.start_step = n_step
            timestep, feed_dict = self.reset_rollout()

            # Get the action and step in the environment
            action_output_np = sess.run(action_output, feed_dict)
            timestep, feed_dict = self.step_rollout(action_output_np)

            self.rew_list.append(timestep.reward)
            self.latent_list.append(action_output_np["latent"].flatten())
            self.prior_mean.append(action_output_np["prior_mean"].flatten())
            # self.latent_noise.append(action_output_np["latent_noise"].flatten())
            self.level_1_scale.append(action_output_np["level_1_scale"].flatten())
            self.level_1_loc.append(action_output_np["level_1_loc"].flatten())
            self.latent_sample.append(action_output_np["latent_sample"].flatten())
            self.latent_mean_list.append(action_output_np["latent_mean"].flatten())

            # Save a checkpoint of the data and video
            if n_step + 1 == self.video_length:
                self.checkpoint_eval_video(
                    os.path.join(self.save_dir, "video", str(self.start_step) + ".mp4")
                )
                self.save_checkpoint(
                    os.path.join(self.save_dir, "logs", str(self.start_step) + ".mat")
                )

                # Clear the cam list to keep memory low.
                self.cam_list = []

    def save_checkpoint(self, save_path: Text):
        """Save data checkoint.

        Args:
            save_path (Text): Directory in which to save output.
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        action_names = self.environment.physics.named.data.act.axes.row.names
        savemat(
            save_path,
            {
                "level_1_scale": np.array(self.level_1_scale),
                "level_1_loc": np.array(self.level_1_loc),
                "latent_sample": np.array(self.latent_sample),
                "reward": np.array(self.rew_list),
                "action_mean": np.array(self.action_mean),
                "action_names": action_names,
                "jacobian_latent_mean": np.array(self.jacobian_latent_mean),
                "walker_body_sites": np.array(self.walker_body_sites),
            },
        )


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
        dest="import_dir",
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

    Args:
        ref_path (Text): Path to .hdf5 reference trajectories.
        save_dir (Text): Folder in which to save .mat files.
        dataset (Text): Name of dataset registered in dm_control.
        import_dir (Text): Path to rodent tracking model.

    Optional Args:
        stac_params (Text): Path to stac params (.yaml).
        offset_path (Text): Path to stac output with offset(.p).
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
        "import_dir",
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
    args = parser.parse_args()
    # Load in parameters to modify
    with open("_batch_args.p", "rb") as file:
        batch_args = pickle.load(file)
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # task_id = 0
    batch_args = batch_args[task_id]
    print(batch_args)
    npmp = NpmpEmbedder(**args.__dict__, **batch_args)
    npmp.embed()


def npmp_embed():
    """Embed entire clip."""
    args = parse()
    npmp = NpmpEmbedder(**args.__dict__)
    npmp.embed()
