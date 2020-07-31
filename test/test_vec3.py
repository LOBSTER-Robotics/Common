import math

import unittest

import numpy as np

from lobster_common.quaternion import Quaternion
from lobster_common.vec3 import Vec3


class Vec3Test(unittest.TestCase):

    def test_rotate(self):
        # Added seed to make tests with np.random deterministic
        np.random.seed(0)

        for _ in range(1000):
            vec = Vec3(np.random.rand(3))

            rotation = Quaternion.from_euler(Vec3(np.random.rand(3) * 4 * math.pi - 2 * math.pi))

            numpy_method = Vec3(rotation.get_rotation_matrix().dot(vec.numpy()))

            rotate_method = vec.rotate(rotation)

            self.assertEqual(numpy_method, rotate_method)
