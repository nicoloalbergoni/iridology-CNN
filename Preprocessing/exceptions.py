class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class CircleNotFoundError(Error):
    def __init__(self, message):
        self.message = message


class ConfigurationFileNotFoundError(Error):
    def __init__(self, message):
        self.message = message

class WrongRadius(Error):
    def __init__(self, message):
        self.message = message