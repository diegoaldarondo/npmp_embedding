from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import dispatch_embed
import embed


def set_up_experiment(params):
    system = embed.System(
        ref_path=params["ref_path"],
        model_dir=params["model_dir"],
        dataset=params["dataset"],
        stac_params=params["stac_params"],
        offset_path=params["offset_path"],
        start_step=0,
        torque_actuators=params["torque_actuators"],
    )
    if params["lstm"]:
        observer = embed.LstmObserver(system.environment, params["save_dir"])
        feeder = embed.LstmFeeder()
    else:
        observer = embed.MlpObserver(system.environment, params["save_dir"])
        feeder = embed.MlpFeeder()
    loop = embed.ClosedLoop(system.environment, feeder, start_step=0, video_length=5)
    return embed.Experiment(system, observer, loop)


def change_exp_model(exp):
    is_mlp = isinstance(exp.observer, embed.MlpObserver)
    if is_mlp:
        exp.system.model_dir = params[1]["model_dir"]
        exp.observer.setup_model_ovservables(embed.LSTM_DATA_TYPES)
        exp.loop.feeder = embed.LstmFeeder()
    else:
        exp.system.model_dir = params[0]["model_dir"]
        exp.observer.setup_model_ovservables(embed.MLP_DATA_TYPES)
        exp.loop.feeder = embed.MlpFeeder()
    return exp


# TODO: figure out how to cleanly setup dm_control environment
# without camera namspace conflicts.
# Hack to avoid problems with overlapping camera namespace.
params = dispatch_embed.build_params("test_params.yaml")

# Test MLP
# EXP = set_up_experiment(params[0])

# Test LSTM
EXP = set_up_experiment(params[1])


class ExperimentTest(absltest.TestCase):
    def test_setup(self):
        self.assertTrue(isinstance(EXP, embed.Experiment))

    def test_run_mlp(self):
        EXP.run()


class ObserverTest(absltest.TestCase):
    def clear_observations(self):
        EXP.observer.cam_list = []

    def setUp(self):
        self.clear_observations()

    def tearDown(self):
        self.clear_observations()

    def test_grab_frame_no_segmentation_mlp(self):
        self.grab_frame(EXP, False)

    def test_grab_frame_segmentation_mlp(self):
        self.grab_frame(EXP, True)

    def grab_frame(self, experiment, seg_frames):
        experiment.observer.seg_frames = seg_frames
        experiment.observer.grab_frame()
        self.assertEqual(experiment.observer.cam_list[0].shape, tuple(embed.IMAGE_SIZE))


class LoopTest(absltest.TestCase):
    def loop(self, loop_fn, experiment):
        experiment.loop = loop_fn(
            experiment.system.environment,
            experiment.loop.feeder,
            experiment.loop.start_step,
            experiment.loop.video_length,
        )
        experiment.run()

    def test_open(self):
        self.loop(embed.OpenLoop, EXP)

    def test_closed(self):
        self.loop(embed.ClosedLoop, EXP)

    # def test_open_lstm(self):
    #     self.loop(embed.OpenLoop, lstm_exp)

    # def test_closed_lstm(self):
    #     self.loop(embed.ClosedLoop, lstm_exp)


if __name__ == "__main__":
    absltest.main()
