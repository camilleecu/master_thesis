import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pct.parser.settings import Settings 
from pct.parser.exceptions import (
    InvalidPercentageException,
    InvalidValuesException,
    MissingMandatoryValueException,
    MissingSectionException,
    OptionNotAvailableException,
    PositiveValueRequiredException,
    WrongValueLengthException
)

class Parser:

    sectionNames = [
        'General','Data','Attributes','Output','Model','Tree','Hierarchical','Forest'
    ]
    mandatoryAttributes = {
        'Data'      :['File'],
        'Attributes':['Target']
    }
    possibleValues = {
        'Verbose':[0,1],
        'Type':[None, 'Tree', 'DAG'],
        'WType':['ExpSumParentWeight','ExpAvgParentWeight','ExpMinParentWeight','ExpMaxParentWeight','NoWeight']
    }

    def __init__(self, filename):
        self.settings = Settings(filename)
        self.lineCounter = 0
        file = open(filename,'r')
        self.lines = file.readlines()	
        self.fileSize = len(self.lines)
        while self.lineCounter < self.fileSize:
            name = self.parseSectionName(self.getLine())
            self.parseSection(name)
        self.checkMandatoryAttributes()
        self.checkAttributeValues()

    def parseSectionName(self, name):
        sectionName = name.strip()[1:-1] 
        if sectionName not in self.sectionNames:
            raise MissingSectionException(sectionName,str(self.sectionNames))
        return sectionName

    def getLine(self):
        """Reads the current non-empty, non-comment (%) line from the settingsfile."""
        if self.lineCounter < self.fileSize:
            line = self.lines[self.lineCounter].strip()
            while len(line) == 0 or line.startswith("%"):
                self.increaseLineCounter()
                if self.lineCounter == self.fileSize:
                    # Gimmick for files ending in an empty line or comment
                    return None
                line = self.lines[self.lineCounter].strip()
            return self.lines[self.lineCounter]
        return None

    def increaseLineCounter(self):
        self.lineCounter+=1

    def parseSection(self, name):
        section = self.settings.sections[name]
        self.increaseLineCounter()
        while not self.isSectionName(self.getLine()):
            line = self.getLine()
            parsedLine = self.parseLine(line,name)
            self.increaseLineCounter()
            section.setAttribute(parsedLine[0],parsedLine[1])
        # section.toString()

    def parseLine(self, line, sectionName):
        splitLine = [l.strip() for l in line.split('=')]
        splitLine[1] = self.convertStringValue(splitLine[1]) 
        return (splitLine[0],splitLine[1])

    def isSectionName(self, line):
        if line is None:
            return True
        line = line.strip()
        return line.startswith('[') and line.endswith(']')

    @staticmethod
    def convertStringValue(value):
        """Parses the given value into a Python type."""
        if value.lower() == 'none':
            return None

        # Booleans
        if value.lower() in ['false', 'no']:
            return False
        elif value.lower() in ['true', 'yes']:
            return True
        
        # List of values -- TODO what about lists of string values?
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1].split(',')
            for i in range(len(value)):
                value[i] = Parser.convertStringToNumber(value[i])
        
        # Regular numbers
        toNumber = Parser.convertStringToNumber(value)
        if toNumber is not None:
            return toNumber
        
        # String values
        else:
            return value

    @staticmethod
    def convertStringToNumber(value):
        """Converts the given value to an integer or float.
        Returns None if the given value is not a number.
        """
        try:
            floatSuccess = False
            floatTry = float(value)
            floatSuccess = True
            intTry = int(value)
            if floatTry == intTry:
                return intTry
            else:
                return floatTry
        except: 
            # TODO This should never run right? int() on a float should round down
            # However, the try clause is very good, it helps on string values
            if floatSuccess:
                return floatTry

    def checkMandatoryAttributes(self):
        for key in self.mandatoryAttributes:
            section = self.settings.sections[key]
            for attr in self.mandatoryAttributes[key]:
                if not section.attributes[attr].enabled:
                    raise MissingMandatoryValueException(attr,key)
                elif attr in self.possibleValues and section.attributes[attr].value not in self.possibleValues[attr]:
                    raise OptionNotAvailableException(attr,section.attributes[attr].value,self.possibleValues[attr])

    def checkAttributeValues(self):
        # General checks
        pass

        # Data checks
        filename  = self.settings.sections["Data"].attributes["File"].value
        testset   = self.settings.sections["Data"].attributes["TestSet"].value
        hierarchy = self.settings.sections["Data"].attributes["Hierarchy"].value
        basePath = Path(self.settings.filename).parent.absolute() # Path where settings file is located
            # (should be equivalent to Path(os.getcwd(), self.settings.filename))
            # This basePath allows us to construct an absolute path from a given path
            # The given path may be both relative and absolute
        if True: # Always do this for training dataset
            filename = Path(basePath, filename) # This is how such an absolute path is created
            if not filename.exists():
                raise FileNotFoundError(filename)
            self.settings.sections["Data"].attributes["File"].value = str(filename)
        if testset != "": # Only do this for testset if one is given
            testset  = Path(basePath, testset)
            if not testset.exists():
                raise FileNotFoundError(testset)
            self.settings.sections["Data"].attributes["TestSet"].value = str(testset)
        if hierarchy != "": # If a hierarchy file was given, check if it exists
            hierarchy  = Path(basePath, hierarchy)
            if not hierarchy.exists():
                raise FileNotFoundError(hierarchy)
            self.settings.sections["Data"].attributes["Hierarchy"].value = str(hierarchy)
        else: # If no hierarchy file was given, check if one should've been given
            hierType = self.settings.sections["Hierarchical"].attributes["Type"].value
            if hierType is not None:
                # TODO better exception for this? (only mandatory when hierType is not None)
                raise MissingMandatoryValueException("Hierarchy","Data")
        
        # Attribute checks
        targets = self.settings.sections["Attributes"].attributes["Target"].value
        data = pd.read_csv(filename)
        if min(targets) < 0:
            raise PositiveValueRequiredException("Target", min(targets))
        if max(targets) > len(data.columns):
            raise InvalidValuesException(
                "Target",targets, "number of columns in the data",len(data.columns)
            )
        # set [Attributes]Clustering and [Attributes]Descriptive here?
        # Clustering should be equal to Target
        # Descriptive should be the remaining columns, unless semi-supervised (then all columns)

        # Model checks
        minWeight = self.settings.sections["Model"].attributes["MinimalWeight"].value
        if minWeight < 0:
            raise PositiveValueRequiredException("MinimalWeight", minWeight)

        # Tree checks
        FTest = self.settings.sections["Tree"].attributes["FTest"].value
        if FTest <= 0:
            raise PositiveValueRequiredException("FTest", FTest)

        # Hierarchical checks
        hierType = self.settings.sections["Hierarchical"].attributes["Type"].value
        if hierType is not None:
            wparam   = self.settings.sections["Hierarchical"].attributes["WParam"].value
            classThr = self.settings.sections["Hierarchical"].attributes["ClassificationThreshold"].value
            if wparam <= 0:
                raise PositiveValueRequiredException("WParam", wparam)
            if min(classThr) < 0:
                raise PositiveValueRequiredException("ClassificationThreshold", min(classThr))
            if max(classThr) > 100:
                raise InvalidPercentageException("ClassificationThreshold/100", max(classThr)/100)
        
        # Forest checks
        numTrees = self.settings.sections["Forest"].attributes["NumberOfTrees"].value
        if numTrees is not None:
            if numTrees <= 0:
                raise PositiveValueRequiredException("NumberOfTrees", numTrees)
