import math
import unittest

import numpy as np

from lobster_common import vec3, quaternion
from lobster_common.third_party import transformations
from lobster_common.vec3 import Vec3


class QuaternionTest(unittest.TestCase):

    def test_from_euler(self):

        eulers = [vec3.Vec3([0.37239115293379754, 0.8885675365428284, 0.07034164404370014]),
                  vec3.Vec3([2.240170194417122, 4.872254082316221, 3.7033353480320415]),
                  vec3.Vec3([5.1517486293576615, 5.1507259244611765, 0.4584012635565024])]

        quaternions = [
            quaternion.Quaternion([0.152193901942652, 0.42799864063339793, -0.048317065922423776, 0.8895614880198457]),
            quaternion.Quaternion([-0.08141833536599752, -0.736720991554386, -0.15685523583263764, 0.6526941628827703]),
            quaternion.Quaternion([-0.3376418811243629, -0.543809971650204, -0.11816370915792139, 0.7591482152272033])]

        for i in range(len(eulers)):
            euler = eulers[i]
            q = quaternions[i]
            calculated_quaternion = quaternion.Quaternion.from_euler(euler_angles=euler)

            ratios = calculated_quaternion.numpy() / q.numpy()

            for ratio in ratios:
                self.assertAlmostEqual(1.0, ratio, places=6)

    def test_from_matrix(self):
        np.random.seed(0)
        for _ in range(100):
            q = quaternion.Quaternion(np.random.rand(4))

            matrix = q.get_rotation_matrix()

            q2 = quaternion.Quaternion.from_rotation_matrix(matrix)

            self.assertTrue(q.almost_equal(q2))

    def test_almost_equal(self):
        np.random.seed(0)
        for _ in range(100):
            q_np = np.random.rand(4)
            q2_np = q_np + np.array([10e-9, 10e-9, 10e-9, 10e-9])

            q1 = quaternion.Quaternion(q_np)
            q2 = quaternion.Quaternion(q2_np)

            self.assertTrue(q1.almost_equal(q2))

        for _ in range(100):
            q_np = np.random.rand(4)
            q2_np = q_np * np.array([1 + 10e-5, 1 + 10e-5, 1 + 10e-5, 1 + 10e-5])

            q1 = quaternion.Quaternion(q_np)
            q2 = quaternion.Quaternion(q2_np)

            self.assertTrue(q1.almost_equal(q2))
