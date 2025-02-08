class InvalidPercentageException(Exception):
    def __init__(self, attrName, attrValue):
        message = f'Attribute "{attrName}" should contain a percentage between 0 and 1, got {attrValue} instead.'
        super().__init__(message)

class InvalidValuesException(Exception):
    def __init__(self, attrName1, attrValue1, attrName2, attrValue2):
        message  = f'Attribute "{attrName1}" (= {attrValue1}) should not contain'
        message += f' a higher value than "{attrName2}" (= {attrValue2})'
        super().__init__(message)

class MissingAttributeException(Exception):
    def __init__(self, attributeName, sectionName, availableAttributes):
        message  = f'Attribute "{attributeName}" not valid in section "{sectionName}"'
        message += f', the available attributes are: {availableAttributes}'
        super().__init__(message)

class MissingMandatoryValueException(Exception):
    def __init__(self, attribute, section):
        message = f'Missing value for mandatory attribute "{attribute}" in section "{section}".'
        super().__init__(message)

class MissingSectionException(Exception):
    def __init__(self, sectionName, allSections):
        message = f'Section "{sectionName}" not found, available sections: {allSections}.'
        super().__init__(message)

class OptionNotAvailableException(Exception):
    def __init__(self, attr, value, available):
        message = f'Value "{value}" not valid for attribute "{attr}", available values are: {available}.'
        super().__init__(message)

class PositiveValueRequiredException(Exception):
    def __init__(self, attrName, attr):
        message = f'A positive value is expected for attribute "{attrName}", got {attr} instead.'
        super().__init__(message)

class WrongValueException(Exception):
    def __init__(self, attr, value, expected):
        message = f'Wrong value for attribute "{attr}": expected {expected}, got {type(value)} instead.'
        super().__init__(message)

class WrongValueLengthException(Exception):
    def __init__(self, attrName, attr,expected):
        message = f'Wrong number of values for attribute "{attr}": expected {expected}, got {len(attr)} instead.'
        super().__init__(message)