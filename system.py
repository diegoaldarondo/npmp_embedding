from dm_control.locomotion.walkers import rodent
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control import composer
from dm_control.suite.wrappers import action_scale
import pickle
import numpy as np
import pickle
import yaml
import tensorflow.compat.v1 as tf

# import tf.contrib as ge
from typing import Dict, Text, Tuple

LATENT_DIM = (1, 60)
NOISE_FLOOR = 0.01
SCALE_MULTIPLIER = 0.99


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
        latent_noise: Text = None,
        noise_gain: float = 1.0,
    ):
        """Initialize system for rat imitation experiments.

        Args:
            ref_path (Text): Path to reference snippet.
            model_dir (Text): Directory of trained model.
            dataset (Text): Name of dataset registered in dm_control.
            stac_params (Text): Path to stack params.yaml file.
            offset_path (Text, optional): Path to offsets .pickle file. Defaults to None.
            arena (composer.Arena, optional):  Arena in which to perform roll out. Defaults to floors.Floor(size=(10.0, 10.0)).
            ref_steps (Tuple, optional): Reference steps. Defaults to (1, 2, 3, 4, 5).
            termination_error_threshold (float, optional): Error threshold at which to stop roll out.. Defaults to 0.25.
            min_steps (int, optional): Minimum number of steps in a roll out. Defaults to 10.
            reward_type (Text, optional): Type of reward. Defaults to "rat_mimic_force".
            physics_timestep (float, optional): Timestep for physics calculations. Defaults to 0.001.
            body_error_multiplier (float, optional): Scaling factor for body error. Defaults to 10.
            video_length (int, optional): Length of snippet in frames. Defaults to 2500.
            min_action (float, optional): Minimum value of action. Defaults to -1.0.
            max_action (float, optional): Maximum value of action. Defaults to 1.0.
            start_step (int, optional): First step of video. Defaults to 0.
            torque_actuators (bool, optional): If True, use torque as model output.
                Defaults to False.
            latent_noise (Text, optional): Type of latent noise. Defaults to None.
            noise_gain (float, optional): Scaling factor for latent noise. Defaults to 1.0.
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
        self.latent_noise = latent_noise
        self.noise_gain = noise_gain

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

    def load_model(self, sess: tf.Session, **kwargs):
        """Load comic model

        Args:
            sess (tf.Session): Tensorflow Session
        """
        zeros_tensor_name = (
            "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/zeros_like_1:0"
        )
        gaussian_tensor = tf.random.normal(LATENT_DIM) * self.noise_gain
        sigmoid_name = (
            "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/Sigmoid:0"
        )
        placeholder = tf.placeholder_with_default(
            tf.ones(LATENT_DIM, dtype=tf.float32),
            shape=LATENT_DIM,
            name="placeholder",
        )
        if self.latent_noise in ["standard", "uniform", "inverted"]:
            tf.saved_model.loader.load(
                sess,
                ["tag"],
                self.model_dir,
                input_map={
                    zeros_tensor_name: gaussian_tensor,
                    sigmoid_name: placeholder,
                },
                **kwargs
            )
        elif self.latent_noise is None or self.latent_noise == "none":
            tf.saved_model.loader.load(sess, ["tag"], self.model_dir, **kwargs)
        else:
            raise ValueError("Unknown latent noise type.")
