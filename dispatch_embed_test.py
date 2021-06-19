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
        for batch_params in params:
            time.sleep(1)
            dispatch_embed.dispatch(batch_params)


if __name__ == "__main__":
    absltest.main()
