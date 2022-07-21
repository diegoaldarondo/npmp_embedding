import tensorflow.compat.v1 as tf
from typing import Dict, Tuple, Callable
import abc
import observer
import feeder
import numpy as np

# Weighting and noise floor for latent scale in model.
WEIGHT = 0.99
FLOOR = 0.01


class Loop(abc.ABC):
    def __init__(
        self,
        env,
        feeder: feeder.Feeder,
        start_step: int,
        video_length: int,
        closed: bool = True,
        action_noise: bool = False,
    ):
        self.env = env
        self.feeder = feeder
        self.start_step = start_step
        self.video_length = video_length
        self.closed = closed
        self.action_noise = action_noise

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
        if self.action_noise:
            timestep = self.env.step(action_output_np["action"])
        else:
            timestep = self.env.step(action_output_np["action_mean"])
        feed_dict = self.feeder.feed(timestep, action_output_np)
        return timestep, feed_dict

    def loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
        observer: observer.Observer = observer.NullObserver(),
    ):
        """Roll-out the model in closed loop with the environment.

        Args:
            sess (tf.Session): Tensorflow session
            action_output (Dict): Dictionary of logged outputs
            timestep (TYPE): Timestep object for the current roll out.
            feed_dict (Dict): Dictionary of inputs
        """
        # TODO: appropriate handling of end of loop without try
        try:
            for n_step in range(self.video_length):
                print(n_step, flush=True)
                observer.grab_frame()

                # Restart at the new step tf the task failed or in open loop
                if (self.closed and timestep.last()) or not self.closed:
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
            self.end_loop(observer)

    def end_loop(self, observer: observer.Observer):
        """Handle the end of the recording.

        Args:
            observer (observer.Observer): Experiment observer
        """
        while len(observer.data["reward"]) < self.video_length:
            for data_type in observer.data.keys():
                observer.data[data_type].append(observer.data[data_type][-1])
        while len(observer.cam_list) < self.video_length:
            observer.cam_list.append(observer.cam_list[-1])
        observer.checkpoint(str(self.start_step))

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
        super().__init__(*args, **kwargs, closed=True)


class OpenLoop(Loop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, closed=False)


class ClosedLoopMultiSample(ClosedLoop):
    def __init__(self, *args, n_samples: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
        observer: observer.Observer = observer.NullObserver(),
    ):
        """Roll-out the model in closed loop with the environment.

        Args:
            sess (tf.Session): Tensorflow session
            action_output (Dict): Dictionary of logged outputs
            timestep (TYPE): Timestep object for the current roll out.
            feed_dict (Dict): Dictionary of inputs
        """
        # TODO: appropriate handling of end of loop without try
        try:
            for n_step in range(self.video_length):
                print(n_step, flush=True)
                observer.grab_frame()

                # Restart at the new step tf the task failed or in open loop
                if (self.closed and timestep.last()) or not self.closed:
                    self.env.task.start_step = n_step
                    timestep, feed_dict = self.reset()

                # Resample n_samples times
                for n_sample in range(self.n_samples):

                    # Get all of the data

                    action_output_np = sess.run(action_output, feed_dict)
                    if n_sample == 0:
                        # Make observations
                        observer.observe(action_output_np, timestep)
                    else:
                        # Retain the sampled action_means
                        observer.data["action_mean"].append(
                            action_output_np["action_mean"].copy()
                        )

                timestep, feed_dict = self.step(action_output_np)

                # Save a checkpoint of the data and video
                if n_step + 1 == self.video_length:
                    observer.checkpoint(str(self.start_step))
        except IndexError:
            self.end_loop(observer)

    def end_loop(self, observer: observer.Observer):
        """Handle the end of the recording.

        Args:
            observer (observer.Observer): Experiment observer
        """
        while len(observer.data["reward"]) < self.video_length:
            for data_type in observer.data.keys():
                observer.data[data_type].append(observer.data[data_type][-1])
        # Add the rest of the action mean samples to round out the data
        while len(observer.data["action_mean"]) < self.video_length * self.n_samples:
            observer.data["action_mean"].append(observer.data["action_mean"][-1])
        while len(observer.cam_list) < self.video_length:
            observer.cam_list.append(observer.cam_list[-1])
        observer.checkpoint(str(self.start_step))


class ClosedLoopOverwriteLatents(ClosedLoop):
    def __init__(
        self,
        env,
        feeder: feeder.Feeder,
        start_step: int,
        video_length: int,
        overwrite_fn: Callable,
        **kwargs
    ):
        super().__init__(env, feeder, start_step, video_length, **kwargs)
        self.overwrite_fn = overwrite_fn

    def loop(
        self,
        sess: tf.Session,
        action_output: Dict,
        timestep,
        feed_dict: Dict,
        observer: observer.Observer = observer.NullObserver(),
    ):
        """Roll-out the model in closed loop with the environment.

        Args:
            sess (tf.Session): Tensorflow session
            action_output (Dict): Dictionary of logged outputs
            timestep (TYPE): Timestep object for the current roll out.
            feed_dict (Dict): Dictionary of inputs
        """
        # TODO: appropriate handling of end of loop without try
        try:
            for n_step in range(self.video_length):
                print(n_step, flush=True)
                observer.grab_frame()

                # Restart at the new step tf the task failed or in open loop
                if (self.closed and timestep.last()) or not self.closed:
                    self.env.task.start_step = n_step
                    timestep, feed_dict = self.reset()

                # Feed the new values into the placeholder
                feed_dict["placeholder:0"] = self.overwrite_fn(sess, feed_dict)

                # Run with the new values
                action_output_np = sess.run(action_output, feed_dict)
                timestep, feed_dict = self.step(action_output_np)

                # Make observations
                observer.observe(action_output_np, timestep)

                # Save a checkpoint of the data and video
                if n_step + 1 == self.video_length:
                    observer.checkpoint(str(self.start_step))
        except IndexError:
            self.end_loop(observer)


def standard_noise(sess: tf.Session, feed_dict: Dict) -> np.ndarray:
    """Get the standard latent noise after the sigmoid operation

    Args:
        sess (tf.Session): Tensorflow session
        feed_dict (Dict): Dictionary of inputs

    Returns:
        np.ndarray: standard latent noise after sigmoid
    """
    X = sess.run(
        sess.graph.get_tensor_by_name(
            "agent_0/step_1/reset_core_1/MultiLevelSamplerWithARPrior/split:1"
        ),
        feed_dict,
    )
    sigmoid = 1 / (1 + np.exp(-X))
    return sigmoid


def invert_noise(sess: tf.Session, feed_dict: Dict) -> np.ndarray:
    """Change the latent noise to be inversely distributed across the latent space.

    Args:
        sess (tf.Session): Tensorflow session
        feed_dict (Dict): Dictionary of inputs

    Returns:
        np.ndarray: Inverted noise
    """
    sigmoid = standard_noise(sess, feed_dict)

    # Make the large values equal the small ones and vice versa in order.
    inds = np.argsort(sigmoid, axis=1).flatten()
    sigmoid[:, inds] = sigmoid[:, inds[::-1]]
    return sigmoid


def uniform_noise(sess: tf.Session, feed_dict: Dict) -> np.ndarray:
    """Change latent noise to be uniformly distributed across the latent space.

    Args:
        sess (tf.Session): Tensorflow session
        feed_dict (Dict): Dictionary of inputs

    Returns:
        np.ndarray: Uniform noise
    """
    sigmoid = standard_noise(sess, feed_dict)

    # Compute the noise for all latent dimensions that produces a ND-Gaussian with variance equal
    # to the variance of the original ND Gaussian, accouting for the weighting and floor terms.
    # Assumes the Gaussian dimensions are independent.
    uniform_average = (
        np.ones_like(sigmoid)
        * (np.sqrt(np.mean((WEIGHT * sigmoid + FLOOR) ** 2, axis=1)) - FLOOR)
        / WEIGHT
    )
    return uniform_average


def clamp_noise(
    sess: tf.Session,
    feed_dict: Dict,
    noise_type,
) -> np.ndarray:
    """Clamp latent noise

    Clamps latent variability to equal the variance of an ND Gaussian times the noise gain.

    Rescales the latent scale such that:

    sum_N(var((a*f(z_i)*w+b)*g*N(0,1))) = sum_N(var(g*N(0,1)))
    f: latent noise distribution function
    z_i: ith latent scale
    N: n dimensions
    w: weighting
    b: noise floor
    g: noise_gain
    a: rescaling factor.

    solves for a and returns f(z)*a

    Args:
        sess (tf.Session): Tensorflow session
        feed_dict (Dict): Dictionary of inputs
        noise_type (str): Type of noise to clamp. ["uniform", "standard", "inverted"]

    Returns:
        np.ndarray: Rescaled sigmoid activation
    """
    sigmoid = get_noise_fn(noise_type)(sess, feed_dict)
    a = WEIGHT**2 * np.sum(sigmoid**2, axis=1)
    b = 2 * WEIGHT * FLOOR * np.sum(sigmoid, axis=1)
    c = sigmoid.size * (FLOOR**2 - 1)
    multiplier = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return sigmoid * multiplier


def get_noise_fn(noise_type: str) -> Callable:
    """Get the noise function.

    Args:
        noise_type (str): Type of noise to clamp. ["uniform", "standard", "inverted"]

    Raises:
        ValueError: If the noise type is not recognized.

    Returns:
        Callable: Noise function
    """
    if noise_type == "standard":
        return standard_noise
    elif noise_type == "inverted":
        return invert_noise
    elif noise_type == "uniform":
        return uniform_noise
    else:
        raise ValueError(
            "Noise type must be one of ['standard', 'inverted', 'uniform']"
        )
