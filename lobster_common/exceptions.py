class ArgumentNoneError(ValueError):
    """
    Raised when an argument of a function was None but shouldn't be.
    """
    pass


class ArgumentLengthError(ValueError):
    """
    Raised when an argument of a function was not the right length.
    """
    pass


class InputDimensionError(ValueError):
    """
    Raised when the size of the input dimension is incorrect
    """
