import math
import unittest
import numpy as np

from lobster_common.quaternion import Quaternion
from lobster_common.vec3 import Vec3


class CoordinateSystemConversionTest(unittest.TestCase):

    def test_conversion(self):
        # Added seed to make tests with np.random deterministic
        np.random.seed(0)

        for _ in range(50):
            # Creating random vector
            vec = Vec3(np.random.rand(3))

            # Creating random rotation
            q = Quaternion.from_euler(Vec3(np.random.rand(3) * 2 * math.pi))

            # Rotate the vector by the rotation
            rotated_vec = vec.rotate(q)

            # transform the original vector to the NED coordinate system
            vec_NED = Vec3.fromENU(vec)

            # transform the original quaternion to the NED coordinate system
            q_NED = Quaternion.fromENU(q)

            # Rotate the converted vector by the converted rotation
            rotated_vec_NED = vec_NED.rotate(q_NED)

            # transform the rotated vector to the NED coordinate system
            rotated_vec_NED_transformed = Vec3.fromENU(rotated_vec)

            np.testing.assert_allclose(rotated_vec_NED.numpy(), rotated_vec_NED_transformed.numpy())
