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

            numpy_method_inv = vec3.Vec3(rotation.get_inverse_rotation_matrix().dot(vec.numpy()))
            rotate_method_inv = vec.rotate_inverse(rotation)

            self.assertTrue(np.allclose(numpy_method.numpy(), rotate_method.numpy()))
            self.assertTrue(np.allclose(numpy_method_inv.numpy(), rotate_method_inv.numpy()))

    def test_cross_product(self):
        a = vec3.Vec3([1, 2, 3])
        b = vec3.Vec3([4, 5, 6])

        self.assertEquals(a.cross_product(b), vec3.Vec3([-3,  6, -3]))

    def test_magnitude(self):
        np.random.seed(0)

        for _ in range(100):
            vector = vec3.Vec3(np.random.rand(3))

            self.assertAlmostEquals(vector.magnitude(), math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2))

    def test_normalized(self):
        np.random.seed(0)

        for _ in range(100):
            vector = vec3.Vec3(np.random.rand(3))
            copy_vector = vec3.Vec3(vector)

            norm_vector = vector.normalized()

            # Make sure the normalized vector has magnitude 1
            self.assertAlmostEquals(norm_vector.magnitude(), 1)

            # Make sure the original vector has not changed
            self.assertEquals(vector, copy_vector)
