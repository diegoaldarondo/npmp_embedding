from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import os
import time
import dispatch_embed


class DispatchEmbedTest(absltest.TestCase):
    def test_small_job(self):
        params = dispatch_embed.build_params("test_params.yaml")
        # project_folder = "/n/holylfs02/LABS/olveczky_lab/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2/"
        # os.chdir(project_folder)
        # ref_path = "./npmp_preprocessing/total.hdf5"
        # stac_params = "./stac_params/params.yaml"
        # offset_path = "./stac/offset.p"
        # dataset = "dannce_ephys_" + os.path.basename(os.getcwd())
        # is_lstm = [False, True]
        # is_torque_actuators = [False, True]
        # model_dir = [
        #     "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_21380833_1_final",
        #     "/n/holylfs02/LABS/olveczky_lab/Diego/data/dm/comic_models/rodent_tracking_model_24186410_1",
        # ]
        # save_dir = [os.path.join("npmp", os.path.basename(m)) for m in model_dir]
        # for i, (s_dir, model, lstm, torque_actuators) in enumerate(
        #     zip(save_dir, model_dir, is_lstm, is_torque_actuators)
        # ):
        #     time.sleep(1)
        #     batch_file = "_batch_args%d.p" % (i)
        #     dispatch_embed.dispatch(
        #         ref_path,
        #         s_dir,
        #         dataset,
        #         model,
        #         stac_params,
        #         offset_path,
        #         lstm,
        #         torque_actuators,
        #         batch_file,
        #         test=True,
        #     )
        for batch_params in params:
            time.sleep(1)
            dispatch_embed.dispatch(batch_params)


if __name__ == "__main__":
    absltest.main()
