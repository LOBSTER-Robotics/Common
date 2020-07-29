NANOSECONDS_PER_SECOND = int(1e9)
NANOSECONDS_PER_MILLISECOND = int(1e6)
NANOSECONDS_PER_MICROSECOND = int(1e3)


class Time(int):

    def __new__(cls, nanoseconds):
        if cls is Time:
            raise TypeError("Time class cannot be directly instantiated!")
        return int.__new__(cls, nanoseconds)

    def __add__(self, other):
        if isinstance(other, Time):
            return Nanoseconds(super().__add__(other))

        raise TypeError(f"{type(other)} cannot be added to {type(self)}")

    # def __str__(self) -> str:
    #     return f"Time<{super().__str__()[:-9]}.{super().__str__()[-9:]} seconds>"

    @property
    def seconds(self) -> float:
        return self/NANOSECONDS_PER_SECOND

    @property
    def milliseconds(self) -> float:
        return self/NANOSECONDS_PER_MILLISECOND

    @property
    def microseconds(self) -> float:
        return self / NANOSECONDS_PER_MICROSECOND

    @property
    def nanoseconds(self) -> int:
        return int(self)

    def __str__(self) -> str:
        return f"Time<{self/NANOSECONDS_PER_SECOND:.9f} seconds>"


class Seconds(Time):

    def __new__(cls, seconds: float):
        return super().__new__(cls, seconds * NANOSECONDS_PER_SECOND)


class Milliseconds(Time):

    def __new__(cls, milliseconds: float):
        return super().__new__(cls, milliseconds * NANOSECONDS_PER_MILLISECOND)


class Microseconds(Time):
    def __new__(cls, microseconds: float):
        return super().__new__(cls, microseconds * NANOSECONDS_PER_MICROSECOND)


class Nanoseconds(Time):

    def __new__(cls, nanoseconds: int):
        return super().__new__(cls, nanoseconds)



