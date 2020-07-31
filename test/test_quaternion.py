import unittest

from lobster_common import vec3, quaternion


class QuaternionTest(unittest.TestCase):

    def test_from_euler(self):

        eulers = [vec3.Vec3([0.37239115293379754, 0.8885675365428284, 0.07034164404370014]),
                  vec3.Vec3([2.240170194417122, 4.872254082316221, 3.7033353480320415]),
                  vec3.Vec3([5.1517486293576615, 5.1507259244611765, 0.4584012635565024])]

        quaternions = [quaternion.Quaternion([0.152193901942652, 0.42799864063339793, -0.048317065922423776, 0.8895614880198457]),
                       quaternion.Quaternion([-0.08141833536599752, -0.736720991554386, -0.15685523583263764, 0.6526941628827703]),
                       quaternion.Quaternion([-0.3376418811243629, -0.543809971650204, -0.11816370915792139, 0.7591482152272033])]

        for i in range(len(eulers)):
            euler = eulers[i]
            q = quaternions[i]
            calculated_quaternion = quaternion.Quaternion.from_euler(euler_angles=euler)

            ratios = calculated_quaternion.numpy() / q.numpy()

            for ratio in ratios:
                self.assertAlmostEqual(1.0, ratio, places=6)
