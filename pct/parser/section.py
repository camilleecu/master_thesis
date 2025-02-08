from pct.parser.attribute import Attribute
from pct.parser.exceptions import MissingAttributeException

class Section:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
    
    def addAttribute(self, name, value, dtype=None):
        attribute = Attribute(name, value, dtype)
        self.attributes[name] = attribute

    def setAttribute(self, name, value):
        if name not in self.attributes:
            raise MissingAttributeException(name, self.name, list(self.attributes.keys()))
        else:
            self.attributes[name].setValue(value)

    def __str__(self):
        string = f"[{self.name}]\n"
        for attr in self.attributes.values():
            string += str(attr) + "\n"
        return string
