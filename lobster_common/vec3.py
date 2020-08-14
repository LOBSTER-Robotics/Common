from __future__ import annotations

import numbers
from typing import Union, List, Tuple

import numpy as np

from lobster_common.constants import *
from lobster_common.exceptions import InputDimensionError
from lobster_common import quaternion


class Vec3:
    """
    Data class that stores 3 dimensional vectors. The vectors should always be stored in the NED coordinate system, if
    if there ar
    """

    def __init__(self, data: Union[List[float], Tuple[float, float, float], np.ndarray, 'Vec3']):
        """
        Creates a 3 dimensional vector from a data array
        :param data: Array with length 4 in the form [x, y, z]
        """
        assert isinstance(data, np.ndarray) or isinstance(data, List) or isinstance(data, Tuple) or isinstance(data, Vec3)

        if isinstance(data, Vec3):
            data = data.numpy().copy()
            
        self._data: np.ndarray = np.asarray(data)

        assert self._data.shape
        if self._data.shape[0] != 3:
            raise InputDimensionError("A Vec3 needs an input array of length 3")
        elif self._data.dtype != float and self._data.dtype != int:
            raise TypeError(
                f"A Vec3 needs to be instantiated by an array of floats, not an array of {self._data.dtype}")

    def numpy(self) -> np.ndarray:
        return self._data

    @property
    def x(self) -> float:
        return self._data[X]

    @property
    def y(self) -> float:
        return self._data[Y]

    @property
    def z(self):
        return self._data[Z]

    def rotate(self, quaternion: quaternion.Quaternion) -> 'Vec3':
        """
        Rotates the vector by the given quaternion.
        :param quaternion: Rotation
        :return: Rotated vector
        """
        return Vec3(quaternion.get_rotation_matrix().dot(self._data))

    def rotate_inverse(self, quaternion: quaternion.Quaternion) -> 'Vec3':
        """
        Inversely rotates the vector by the given quaternion.
        :param quaternion: Rotation
        :return: Rotated vector
        """
        return Vec3(quaternion.get_inverse_rotation_matrix().dot(self._data))

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self._data + other._data)
        elif isinstance(other, np.ndarray):
            return Vec3(other + self._data)

        raise TypeError(f"A {type(other)} cannot be added to a Vec3")

    def __radd__(self, other):
        if isinstance(other, np.ndarray):
            return Vec3(other + self._data)
        elif isinstance(other, float):
            return Vec3(other + self._data)

        raise TypeError(f"A Vec3 cannot be added to a {type(other)}]")

    def __sub__(self, other) -> 'Vec3':
        if isinstance(other, Vec3):
            return Vec3(self._data - other._data)

        raise TypeError(f"A {type(other)} cannot be subtracted from a Vec3")

    def __rsub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(other._data - self._data)

        raise TypeError(f"A {type(other)} cannot be subtracted from a Vec3")

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Vec3(self._data * other)
        elif isinstance(other, Vec3):
            return Vec3(self._data * other._data)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Vec3(self._data * other)
        elif isinstance(other, Vec3):
            return Vec3(self._data * other._data)

        raise TypeError(f"A Vec3 cannot be multiplied with a {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Vec3(self._data / other)
        elif isinstance(other, Vec3):
            return Vec3(self._data / other._data)

        raise TypeError(f"A Vec3 cannot be divided by a {type(other)}")

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __str__(self):
        return f"Vec3<{self[0]:.4f}, {self[1]:.4f}, {self[2]:.4f}>"

    def __eq__(self, other):
        if isinstance(other, Vec3):
            return (self._data == other._data).all()

        return False

    def asENU(self) -> np.ndarray:
        """
        Transforms the vector to the ENU coordinate system.
        :return: Vector in the ENU coordinate system.
        """
        # Negating the Y and Z axes
        return np.array([self._data[0], -self._data[1], -self._data[2]])

    @staticmethod
    def fromENU(vector: Union['Vec3', List[float], Tuple[float, float, float], np.ndarray]) -> Vec3:
        """
        Takes a vector in from the ENU coordinate system and converts it to the NED coordinate system.
        :param vector: Vector in the ENU coordinate system.
        :return: Vector in the NED coordinate system.
        """
        return Vec3([vector[X], -vector[Y], -vector[Z]])
