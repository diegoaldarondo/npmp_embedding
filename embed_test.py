from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from unittest import TestCase
import dispatch_embed
import experiment
import observer
import unittest
from feeder import LstmFeeder, MlpFeeder
from loop import (
    OpenLoop,
    ClosedLoop,
    ClosedLoopMultiSample,
    ClosedLoopOverwriteLatents,
    uniform_noise,
    invert_noise,
    clamp_noise,
)
from system import System


def set_up_experiment(params):
    system = System(
        ref_path=params["ref_path"],
        model_dir=params["model_dir"],
        dataset=params["dataset"],
        stac_params=params["stac_params"],
        offset_path=params["offset_path"],
        start_step=0,
        # start_step=357500,  # To test the end loop handling
        torque_actuators=params["torque_actuators"],
        latent_noise=params["latent_noise"],
        noise_gain=params["noise_gain"],
    )
    if params["lstm"]:
        obs = observer.LstmObserver(system.environment, params["save_dir"])
        feeder = LstmFeeder()
    else:
        obs = observer.MlpObserver(system.environment, params["save_dir"])
        feeder = MlpFeeder()
    loop = ClosedLoop(
        system.environment, feeder, start_step=0, video_length=2500, action_noise=False
    )
    # loop = ClosedLoopOverwriteLatents(system.environment, feeder, start_step=0, video_length=2500)
    return experiment.Experiment(system, obs, loop)


def change_exp_model(exp):
    is_mlp = isinstance(exp.observer, observer.MlpObserver)
    if is_mlp:
        exp.system.model_dir = params[1]["model_dir"]
        exp.observer.setup_model_ovservables(observer.LSTM_NETWORK_FEATURES)
        exp.looper.feeder = LstmFeeder()
    else:
        exp.system.model_dir = params[0]["model_dir"]
        exp.observer.setup_model_ovservables(observer.MLP_NETWORK_FEATURES)
        exp.looper.feeder = MlpFeeder()
    return exp


# TODO: figure out how to cleanly setup dm_control environment
# without camera namspace conflicts.
# Hack to avoid problems with overlapping camera namespace.
params = dispatch_embed.build_params("test_params.yaml")

# Test MLP
EXP = set_up_experiment(params[0])

# Test LSTM
# EXP = set_up_experiment(params[1])


# class ExperimentTest(absltest.TestCase):
#     def test_setup(self):
#         self.assertTrue(isinstance(EXP, experiment.Experiment))

#     def test_run_mlp(self):
#         EXP.run()


# class ObserverTest(absltest.TestCase):
#     def clear_observations(self):
#         EXP.observer.cam_list = []

#     def setUp(self):
#         self.clear_observations()

#     def tearDown(self):
#         self.clear_observations()

#     def test_grab_frame_no_segmentation_mlp(self):
#         self.grab_frame(EXP, False)

#     def test_grab_frame_segmentation_mlp(self):
#         self.grab_frame(EXP, True)

#     def grab_frame(self, exp, seg_frames):
#         exp.observer.seg_frames = seg_frames
#         exp.observer.grab_frame()
#         self.assertEqual(exp.observer.cam_list[0].shape, tuple(observer.IMAGE_SIZE))


class LoopTest(TestCase):
    def loop(self, loop_fn, exp):
        exp.looper = loop_fn(
            exp.system.environment,
            exp.looper.feeder,
            exp.looper.start_step,
            exp.looper.video_length,
        )
        exp.run()

    # def test_open(self):
    #     self.loop(OpenLoop, EXP)

    def test_closed(self):
        self.loop(ClosedLoop, EXP)

    # def test_closed_loop_overwrite_latents(self):
    #     EXP.looper = ClosedLoopOverwriteLatents(
    #         EXP.system.environment,
    #         EXP.looper.feeder,
    #         EXP.looper.start_step,
    #         EXP.looper.video_length,
    #         lambda sess, feed_dict: clamp_noise(sess, feed_dict, "standard"),
    #         action_noise=True,
    #     )
    #     EXP.run()

    # def test_closed_multi_sample(self):
    #     self.loop(ClosedLoopMultiSample, EXP)

    # def test_end_loop(self):
    # def test_open_lstm(self):
    #     self.loop(experiment.OpenLoop, lstm_exp)

    # def test_closed_lstm(self):
    #     self.loop(experiment.ClosedLoop, lstm_exp)


if __name__ == "__main__":
    unittest.main()
