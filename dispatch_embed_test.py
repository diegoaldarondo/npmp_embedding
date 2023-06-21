from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os
import time
import dispatch_embed


class DispatchEmbedTest(unittest.TestCase):
    def test_small_job(self):
        params = dispatch_embed.build_params("test_params.yaml")
        for batch_params in params:
            time.sleep(1)
            dispatcher = dispatch_embed.ParallelNpmpDispatcher(batch_params)
            dispatcher.dispatch()


if __name__ == "__main__":
    unittest.main()
