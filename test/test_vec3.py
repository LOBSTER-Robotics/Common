import math

import unittest

import numpy as np

from lobster_common import vec3, quaternion


class Vec3Test(unittest.TestCase):

    def test_rotate(self):
        # Added seed to make tests with np.random deterministic
        np.random.seed(0)

        for _ in range(1000):
            vec = vec3.Vec3(np.random.rand(3))

            rotation = quaternion.Quaternion.from_euler(vec3.Vec3(np.random.rand(3) * 4 * math.pi - 2 * math.pi))

            numpy_method = vec3.Vec3(rotation.get_rotation_matrix().dot(vec.numpy()))

            rotate_method = vec.rotate(rotation)

            self.assertEqual(numpy_method, rotate_method)
