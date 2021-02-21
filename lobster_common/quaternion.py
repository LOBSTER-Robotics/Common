from __future__ import annotations

from typing import List, Union, Tuple, Optional

import numpy as np

from lobster_common import vec3
from lobster_common.constants import *
from lobster_common.exceptions import InputDimensionError
from lobster_common.third_party import transformations as trans


class Quaternion:
    """
    Data class that stores quaternions. The quaternions should always be stored in the NED coordinate system
    """

    PRINTING_FORMAT_MINIMAL_WIDTH = -1
    PRINTING_FORMAT_DECIMALS = -1

    def __init__(self, data: Union[List[float], Tuple[float, float, float, float], np.ndarray]):
        """
        Creates a quaternion from a data array
        :param data: Array with length 4 in the form [x, y, z, w]
        """
        assert isinstance(data, np.ndarray) or isinstance(data, List) or isinstance(data, Tuple)

        if isinstance(data, Quaternion):
            data = data.numpy().copy()

        self._data: np.ndarray = np.asarray(data)

        if self._data.shape[0] != 4:
            raise InputDimensionError("A Quaternion needs an input array of length 4")

    def numpy(self) -> np.ndarray:
        return self._data

    def normalized(self):
        return Quaternion(self.numpy() / self.magnitude())

    def magnitude(self):
        return np.linalg.norm(self._data)

    @property
    def x(self) -> float:
        return self._data[X]

    @property
    def y(self) -> float:
        return self._data[Y]

    @property
    def z(self) -> float:
        return self._data[Z]

    @property
    def w(self) -> float:
        return self._data[W]

    def __getitem__(self, key):
        return self._data[key]

    def __str__(self):
        if Quaternion.PRINTING_FORMAT_DECIMALS != -1:
            return f"Quaternion<" \
                   f"x:{self.x:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}.{Quaternion.PRINTING_FORMAT_DECIMALS}f}, " \
                   f"y:{self.y:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}.{Quaternion.PRINTING_FORMAT_DECIMALS}f}, " \
                   f"z:{self.z:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}.{Quaternion.PRINTING_FORMAT_DECIMALS}f}, "\
                   f"w:{self.w:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}.{Quaternion.PRINTING_FORMAT_DECIMALS}f}>"
        elif Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH != -1:
            return f"Quaternion<" \
                   f"x:{self.x:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}f}, " \
                   f"y:{self.y:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}f}, " \
                   f"z:{self.z:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}f}, " \
                   f"w:{self.w:{Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH}f}>"
        else:
            return f"Quaternion<" \
                   f"x:{self.x}, " \
                   f"y:{self.y}, " \
                   f"z:{self.z}, " \
                   f"w:{self.w}>"

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        return Quaternion(trans.quaternion_multiply(self, other))

    def __eq__(self, other: 'Quaternion'):
        return np.equal(self.numpy(), other.numpy()).all()

    def almost_equal(self, other: 'Quaternion') -> bool:
        return np.allclose(self.numpy(), other.numpy())

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Convert input quaternion to 3x3 rotation matrix
        :return: 3x3 rotation matrix.
        """
        n = np.linalg.norm(self.numpy())
        if n == 0.0:
            raise ZeroDivisionError(f"Input to `as_rotation_matrix({self})` has zero norm")

        return np.array([
            [1 - 2 * (self.y ** 2 + self.z ** 2) / n, 2 * (self.x * self.y - self.z * self.w) / n,
             2 * (self.x * self.z + self.y * self.w) / n],
            [2 * (self.x * self.y + self.z * self.w) / n, 1 - 2 * (self.x ** 2 + self.z ** 2) / n,
             2 * (self.y * self.z - self.x * self.w) / n],
            [2 * (self.x * self.z - self.y * self.w) / n, 2 * (self.y * self.z + self.x * self.w) / n,
             1 - 2 * (self.x ** 2 + self.y ** 2) / n]
        ])

    def get_inverse_rotation_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.get_rotation_matrix())

    @staticmethod
    def from_rotation_matrix(matrix: np.ndarray) -> Quaternion:
        if matrix.shape != (3, 3):
            raise ValueError(f"Rotation matrix has to by 3x3 not {matrix.shape}")

        # Add extra column and row to satisfy conditions for transformation
        larger_matrix = np.zeros((4, 4))
        larger_matrix[:-1, :-1] = matrix
        larger_matrix[3, 3] = 1

        return Quaternion(trans.quaternion_from_matrix(larger_matrix))

    def to_euler(self) -> vec3.Vec3:
        """
        Transform to euler [x,y,z]
        """
        return vec3.Vec3(trans.euler_from_quaternion(self))

    @staticmethod
    def from_euler(euler_angles: vec3.Vec3):
        # Figure out the input angles from either type of input
        alpha = euler_angles[X]
        beta = euler_angles[Y]
        gamma = euler_angles[Z]

        # Set up the output array
        R = np.empty(4, dtype=np.double)

        cos = np.cos
        sin = np.sin

        # Compute the actual values of the quaternion components
        # TODO this seems to be a slightly more efficient way to compute this, but produces different results, maybe in
        #  the future we can look at this.
        # R[X] = sin(alpha / 2) * cos((alpha - gamma) / 2)  # x quaternion components
        # R[Y] = sin(beta / 2) * sin((alpha - gamma) / 2)  # y quaternion components
        # R[Z] = cos(beta / 2) * sin((alpha + gamma) / 2)  # z quaternion components
        # R[W] = cos(beta / 2) * cos((alpha + gamma) / 2)  # scalar quaternion components

        R[X] = sin(alpha / 2) * cos(beta / 2) * cos(gamma / 2) - cos(alpha / 2) * sin(beta / 2) * sin(gamma / 2)
        R[Y] = cos(alpha / 2) * sin(beta / 2) * cos(gamma / 2) + sin(alpha / 2) * cos(beta / 2) * sin(gamma / 2)
        R[Z] = cos(alpha / 2) * cos(beta / 2) * sin(gamma / 2) - sin(alpha / 2) * sin(beta / 2) * cos(gamma / 2)
        R[W] = cos(alpha / 2) * cos(beta / 2) * cos(gamma / 2) + sin(alpha / 2) * sin(beta / 2) * sin(gamma / 2)

        return Quaternion(R / np.linalg.norm(R))

    def conjugate(self) -> 'Quaternion':
        """
        Conjugate quaternion.
        """
        return Quaternion(trans.quaternion_conjugate(self))

    def difference(self, other: 'Quaternion') -> Quaternion:
        """
        Get the difference of rotation between two quaternions
        :Quaternion other: Other rotation to compare the difference to.
        :returns: Difference as a quaternion
        """
        # return trans.quaternion_multiply(trans.quaternion_conjugate(self), goal_quat)
        return self.conjugate() * other

    def as_nwu(self) -> np.ndarray:
        """
        Transforms the quaternion to the NWU coordinate system.
        :return: Quaternion as numpy array in the NWU coordinate system.
        """
        # Negating Y and Z
        return np.array([self._data[X], -self._data[Y], -self._data[Z], self._data[W]])

    @staticmethod
    def from_nwu(quaternion: Union[List[float], Tuple[float, float, float, float], np.ndarray]) -> 'Quaternion':
        """
        Creates a quaternion in the NED coordinate system from a given array or Quaternion in the NWU coordinate system
        :param quaternion: Quaternion or array that represents a quaternion
        :return: Quaternion in the NED coordinate system
        """
        # Swapping X and Y and negating Z
        return Quaternion([quaternion[X], -quaternion[Y], -quaternion[Z], quaternion[W]])

    @staticmethod
    def set_printing_format(minimal_width: Optional[int] = None, decimals: Optional[int] = None):
        if minimal_width is not None:
            Quaternion.PRINTING_FORMAT_MINIMAL_WIDTH = minimal_width
        if decimals is not None:
            Quaternion.PRINTING_FORMAT_DECIMALS = decimals
