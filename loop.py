import tensorflow.compat.v1 as tf
from typing import Dict, Tuple
import abc
import observer
import feeder
import numpy as np

_N_REPEATS_NECESSARY_TO_FREEZE_INPUTS = 10


class Loop(abc.ABC):
    def __init__(
        self,
        env,
        feeder: feeder.Feeder,
        start_step: int,
        video_length: int,
        closed: bool = True,
    ):
        self.env = env
        self.feeder = feeder
        self.start_step = start_step
        self.video_length = video_length
        self.closed = closed

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
            # while len(cam_list) < self.video_length:
            #     self.cam_list.append(self.cam_list[-1])
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
                    action_output_np = sess.run(action_output, feed_dict)

                    # Make observations
                    observer.observe(action_output_np, timestep)

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
        while len(observer.data["reward"]) < self.video_length * self.n_samples:
            for data_type in observer.data.keys():
                observer.data[data_type].append(observer.data[data_type][-1])
            # while len(cam_list) < self.video_length:
            #     self.cam_list.append(self.cam_list[-1])
            observer.checkpoint(str(self.start_step))
