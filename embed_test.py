from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import dispatch_embed
import embed


class EmbedTest(absltest.TestCase):
    def setUp(self):
        params = dispatch_embed.build_params("test_params.yaml")
        params = params[0]
        self.npmp = embed.NpmpEmbedder(
            params["ref_path"],
            params["save_dir"],
            params["model_dir"],
            params["dataset"],
            params["stac_params"],
            params["offset_path"],
            lstm=False,
            torque_actuators=True,
            start_step=0,
            end_step=100,
        )

    def test_setup(self):
        self.assertTrue(isinstance(self.npmp, embed.NpmpEmbedder))


if __name__ == "__main__":
    absltest.main()
