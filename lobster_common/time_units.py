NANOSECONDS_PER_SECOND = int(1e9)
NANOSECONDS_PER_MILLISECOND = int(1e6)
NANOSECONDS_PER_MICROSECOND = int(1e3)


class Time(int):
    """
    Data class that stores the time in nanoseconds.
    """

    def __new__(cls, nanoseconds):
        if cls is Time:
            raise TypeError("Time class cannot be directly instantiated!")
        return int.__new__(cls, nanoseconds)

    def __add__(self, other):
        """
        Adds two Time objects together and returns the result in nanoseconds.
        :param other: Time object
        :return: Summed time.
        """

        if isinstance(other, Time):
            return Nanoseconds(super().__add__(other))

        raise TypeError(f"{type(other)} cannot be added to {type(self)}")

    @property
    def seconds(self) -> float:
        """
        Gives the time in seconds.
        :return: Time in seconds.
        """
        return self/NANOSECONDS_PER_SECOND

    @property
    def milliseconds(self) -> float:
        """
        Gives the time in milliseconds.
        :return: Time in milliseconds.
        """
        return self/NANOSECONDS_PER_MILLISECOND

    @property
    def microseconds(self) -> float:
        """
        Gives the time in microseconds.
        :return: Time in microseconds.
        """
        return self / NANOSECONDS_PER_MICROSECOND

    @property
    def nanoseconds(self) -> int:
        """
        Gives the time in nanoseconds.
        :return: Time in nanoseconds.
        """
        return int(self)

    def __str__(self) -> str:
        return f"Time<{self/NANOSECONDS_PER_SECOND:.9f} seconds>"

    def __format__(self, format_spec):
        return str(self)


class Seconds(Time):
    """Creates Time object from float in seconds."""

    def __new__(cls, seconds: float):
        return super().__new__(cls, seconds * NANOSECONDS_PER_SECOND)


class Milliseconds(Time):
    """Creates Time object from float in milliseconds."""

    def __new__(cls, milliseconds: float):
        return super().__new__(cls, milliseconds * NANOSECONDS_PER_MILLISECOND)


class Microseconds(Time):
    """Creates Time object from float in microseconds."""

    def __new__(cls, microseconds: float):
        return super().__new__(cls, microseconds * NANOSECONDS_PER_MICROSECOND)


class Nanoseconds(Time):
    """Creates Time object from float in nanoseconds."""

    def __new__(cls, nanoseconds: int):
        return super().__new__(cls, nanoseconds)
