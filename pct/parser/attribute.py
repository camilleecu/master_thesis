import sys
from pct.parser.exceptions import WrongValueException

class Attribute:
    def __init__(self, name, value, dtype=None):
        self.name = name
        self.value = value
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = type(value)
        self.mandatory = None
        self.enabled = False # Whether user has defined this attribute

    def setValue(self, value):
        if self.checkDataType(value):
            self.value = value
            self.enabled = True
        else:
            raise WrongValueException(self.name, value, self.dtype)

    def checkDataType(self,value):
        # TODO might want to allow subtypes, e.g. passing "FTest = 1" throws an error
        # because 1 is an integer and not a float, but integers are easily converted
        # to floats
        return self.dtype == type(value)

    def __str__(self):
        return f"{self.name} = {self.value}"
