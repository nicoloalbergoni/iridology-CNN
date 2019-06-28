class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class CircleNotFoundError(Error):
    def __init__(self, message):
        self.message = message


class ConfigurationFileNotFoundError(Error):
    def __init__(self, message):
        self.message = message


class CannotLoadImagesError(Error):
    def __init__(self, message):
        self.message = message


class MultipleCirclesFoundError(Error):
    def __init__(self, message):
        self.message = message


class CreateDataError(Error):
    def __init__(self, message):
        self.message = message

