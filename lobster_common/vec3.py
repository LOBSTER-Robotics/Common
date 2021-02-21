from __future__ import annotations

import numbers
from typing import Union, List, Tuple, Optional

import numpy as np

from lobster_common.constants import *
from lobster_common.exceptions import InputDimensionError
from lobster_common import quaternion


class Vec3:
    """
    Data class that stores 3 dimensional vectors. The vectors should always be stored in the NED coordinate system
    """

    PRINTING_FORMAT_MINIMAL_WIDTH = -1
    PRINTING_FORMAT_DECIMALS = -1

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
    def z(self) -> float:
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

    def cross_product(self, other: Vec3) -> Vec3:
        return Vec3(np.cross(self.numpy(), other.numpy()))

    def normalized(self):
        return self / self.magnitude()

    def magnitude(self):
        return np.linalg.norm(self._data)

    def __add__(self, other: Vec3):
        if isinstance(other, Vec3):
            return Vec3(self._data + other._data)

        raise TypeError(f"A {type(other)} cannot be added to a Vec3")

    def __radd__(self, other: Vec3):
        if isinstance(other, Vec3):
            return Vec3(other.numpy() + self.numpy())

        raise TypeError(f"A Vec3 cannot be added to a {type(other)}]")

    def __sub__(self, other: Vec3) -> 'Vec3':
        if isinstance(other, Vec3):
            return Vec3(self._data - other._data)

        raise TypeError(f"A {type(other)} cannot be subtracted from a Vec3")

    def __rsub__(self, other: Vec3):
        if isinstance(other, Vec3):
            return Vec3(other._data - self._data)

        raise TypeError(f"A {type(other)} cannot be subtracted from a Vec3")

    def __mul__(self, other: Union[numbers.Number, Vec3]):
        if isinstance(other, numbers.Number):
            return Vec3(self._data * other)
        elif isinstance(other, Vec3):
            return Vec3(self._data * other._data)

    def __rmul__(self, other: Union[numbers.Number, Vec3]):
        if isinstance(other, numbers.Number):
            return Vec3(other * self._data)
        elif isinstance(other, Vec3):
            return Vec3(other._data * self._data)

        raise TypeError(f"A Vec3 cannot be multiplied with a {type(other)}")

    def __truediv__(self, other: numbers.Number):
        if isinstance(other, numbers.Number):
            return Vec3(self._data / other)

        raise TypeError(f"A Vec3 cannot be divided by a {type(other)}")

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __str__(self):
        if Vec3.PRINTING_FORMAT_DECIMALS != -1:
            return f"Vec3<" \
                   f"{self[0]:{Vec3.PRINTING_FORMAT_MINIMAL_WIDTH}.{Vec3.PRINTING_FORMAT_DECIMALS}f}, " \
                   f"{self[1]:{Vec3.PRINTING_FORMAT_MINIMAL_WIDTH}.{Vec3.PRINTING_FORMAT_DECIMALS}f}, " \
                   f"{self[2]:{Vec3.PRINTING_FORMAT_MINIMAL_WIDTH}.{Vec3.PRINTING_FORMAT_DECIMALS}f}>"
        elif Vec3.PRINTING_FORMAT_MINIMAL_WIDTH != -1:
            return f"Vec3<" \
                   f"{self[0]:{Vec3.PRINTING_FORMAT_MINIMAL_WIDTH}f}, " \
                   f"{self[1]:{Vec3.PRINTING_FORMAT_MINIMAL_WIDTH}f}, " \
                   f"{self[2]:{Vec3.PRINTING_FORMAT_MINIMAL_WIDTH}f}>"
        else:
            return f"Vec3<" \
                   f"{self[0]}, " \
                   f"{self[1]}, " \
                   f"{self[2]}>"

    def __repr__(self):
        return str(self)

    def __eq__(self, other: Vec3):
        if isinstance(other, Vec3):
            return (self._data == other._data).all()

        return False

    def as_nwu(self) -> np.ndarray:
        """
        Transforms the vector to the ENU coordinate system.
        :return: Vector in the ENU coordinate system.
        """
        # Negating Y and Z
        return np.array([self._data[X], -self._data[Y], -self._data[Z]])

    @staticmethod
    def from_nwu(vector: Union['Vec3', List[float], Tuple[float, float, float], np.ndarray]) -> Vec3:
        """
        Takes a vector in from the ENU coordinate system and converts it to the NED coordinate system.
        :param vector: Vector in the ENU coordinate system.
        :return: Vector in the NED coordinate system.
        """
        # Negating Y and Z
        return Vec3([vector[X], -vector[Y], -vector[Z]])

    @staticmethod
    def set_printing_format(minimal_width: Optional[int] = None, decimals: Optional[int] = None):
        if minimal_width is not None:
            Vec3.PRINTING_FORMAT_MINIMAL_WIDTH = minimal_width
        if decimals is not None:
            Vec3.PRINTING_FORMAT_DECIMALS = decimals



