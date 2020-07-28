import math

import unittest

import numpy as np

from Quaternion import Quaternion
from Vec3 import Vec3


class QuaternionTest(unittest.TestCase):

    @staticmethod
    def test_multiply():
        for _ in range(10):
            vec = Vec3(np.random.rand(3))

            rotation = PybulletAPI.getQuaternionFromEuler(Vec3(np.random.rand(3) * 2 * math.pi))

            numpy_method = Vec3(rotation.get_rotation_matrix().dot(vec.numpy()))

            rotate_method = vec.rotate(rotation)

            assert numpy_method == rotate_method

    def test_from_euler(self):

        eulers = [Vec3([-1.262, -1.111, 2.405]),
                  Vec3([-1.316, 0.593, -1.153])]

        quaternions = [Quaternion([-0.217, -0.621, -0.528, 0.537]),
                       Quaternion([-0.616, -0.125, -0.562, 0.537])]

        for i in range(len(eulers)):
            euler = eulers[i]
            quaternion = quaternions[i]
            calculated_quaternion = Quaternion.from_euler(euler_angles=euler)

            ratios = calculated_quaternion.numpy() / quaternion.numpy()
            print(calculated_quaternion, quaternion)

            # for ratio in ratios:
            #     self.assertAlmostEqual(1.0, ratio, places=2)
