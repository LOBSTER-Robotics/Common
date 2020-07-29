from typing import Any, List, Union, Tuple

import numpy as np

from Constants import *
import Vec3
from exceptions import InputDimensionError


class Quaternion:

    def __init__(self, data: Union[List[float], Tuple[float, float, float, float], np.ndarray]):
        """
        Creates a quaternion from a data array
        :param data: Array with length 4 in the form [x, y, z, w]
        """
        assert isinstance(data, np.ndarray) or isinstance(data, List) or isinstance(data, Tuple)

        self._data: np.ndarray = np.asarray(data)

        if self._data.shape[0] != 4:
            raise InputDimensionError("A Quaternion needs an input array of length 4")

    def numpy(self):
        return self._data

    @property
    def x(self):
        return self._data[X]

    @property
    def y(self):
        return self._data[Y]

    @property
    def z(self):
        return self._data[Z]

    @property
    def w(self):
        return self._data[W]

    def asENU(self) -> np.ndarray:
        # Swapping the Y and Z axes
        array = self._data.copy()
        array[1] = -array[1]
        array[2] = -array[2]
        return array

    def __str__(self):
        return f"Quaternion<{self._data}>"

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Convert input quaternion to 3x3 rotation matrix
        :return: 3x3 rotation matrix.
        """
        n = np.linalg.norm(self.numpy())

        if n == 0.0:
            raise ZeroDivisionError(f"Input to `as_rotation_matrix({self})` has zero norm")

        return np.array([
            [1 - 2*(self.y**2 + self.z**2)/n,     2*(self.x*self.y - self.z*self.w)/n, 2*(self.x*self.z + self.y*self.w)/n],
            [2*(self.x*self.y + self.z*self.w)/n, 1 - 2*(self.x**2 + self.z**2)/n,     2*(self.y*self.z - self.x*self.w)/n],
            [2*(self.x*self.z - self.y*self.w)/n, 2*(self.y*self.z + self.x*self.w)/n, 1 - 2*(self.x**2 + self.y**2)/n]
        ])

    def get_inverse_rotation_matrix(self):
        return np.linalg.inv(self.get_rotation_matrix())

    @staticmethod
    def from_euler(euler_angles: Vec3):
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

        R[X] = sin(alpha / 2) * cos(beta / 2) * cos(gamma / 2) - cos(alpha/2) * sin(beta/2)*sin(gamma/2)
        R[Y] = cos(alpha / 2) * sin(beta / 2) * cos(gamma / 2) + sin(alpha/2) * cos(beta/2)*sin(gamma/2)
        R[Z] = cos(alpha / 2) * cos(beta / 2) * sin(gamma / 2) - sin(alpha/2) * sin(beta/2)*cos(gamma/2)
        R[W] = cos(alpha / 2) * cos(beta / 2) * cos(gamma / 2) + sin(alpha/2) * sin(beta/2)*sin(gamma/2)

        return Quaternion(R / np.linalg.norm(R))

    @staticmethod
    def fromENU(quaternion: Union[List[float], Tuple[float, float, float, float], np.ndarray]) -> 'Quaternion':
        """
        Creates a quaternion in the NED coordinate system from a given array or Quaternion in the ENU coordinate system
        :param quaternion: Quaternion or array that represents a quaternion
        :return: Quaternion in the NED coordinate system
        """

        # Conversion follows https://stackoverflow.com/a/18818267, it needs to be checked if this is correct
        if isinstance(quaternion, Quaternion):
            # Swapping the Y and Z axes
            quaternion._data[1] = -quaternion._data[1]
            quaternion._data[2] = -quaternion._data[2]
            return quaternion
        elif isinstance(quaternion, List) or isinstance(quaternion, np.ndarray):
            # Swapping the Y and Z axes
            quaternion[1] = -quaternion[1]
            quaternion[2] = -quaternion[2]
            return Quaternion(quaternion)
        elif isinstance(quaternion, Tuple):
            quaternion: Tuple[float, float, float, float] = (float(quaternion[0]),
                                                             float(-quaternion[1]),
                                                             float(-quaternion[2]),
                                                             float(quaternion[3]))
            return Quaternion(quaternion)

        raise TypeError(f"Can only create NED quaternion from quaternion of array, not {type(quaternion)}")