from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import dispatch_embed
import embed

# TODO: figure out how to cleanly setup dm_control environment
# without camera namspace conflicts.
# Hack to avoid problems with overlapping camera namespace.
params = dispatch_embed.build_params("test_params.yaml")
params = params[0]
NPMP = embed.NpmpEmbedder(
    params["ref_path"],
    params["save_dir"],
    params["model_dir"],
    params["dataset"],
    params["stac_params"],
    params["offset_path"],
    lstm=False,
    torque_actuators=True,
    start_step=0,
    end_step=20,
    video_length=20,
)


class EmbedTest(absltest.TestCase):
    def test_setup(self):
        self.assertTrue(isinstance(NPMP, embed.NpmpEmbedder))

    def test_embed(self):
        NPMP.embed()


class ObserverTest(absltest.TestCase):
    def clear_observations(self):
        NPMP.observer.cam_list = []

    def setUp(self):
        self.clear_observations()

    def tearDown(self):
        self.clear_observations()

    def test_grab_frame_no_segmentation(self):
        NPMP.observer.seg_frames = False
        self.grab_frame()

    def test_grab_frame_segmentation(self):
        NPMP.observer.seg_frames = True
        self.grab_frame()

    def grab_frame(self):
        NPMP.observer.grab_frame()
        self.assertEqual(NPMP.observer.cam_list[0].shape, tuple(embed.IMAGE_SIZE))


if __name__ == "__main__":
    absltest.main()
