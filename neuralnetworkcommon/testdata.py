#!/usr/bin/env python3
# imports
from inspect import signature
from json import loads
from os import linesep
from pythoncommontools.logger import logger
from pythoncommontools.objectUtil.objectUtil import methodArgsStringRepresentation
from pythoncommontools.jsonEncoderDecoder.complexJsonEncoderDecoder import ComplexJsonDecoder
# contants
IMAGE_MARKUP="image"
REPRENSATION_INTERVAL=51.2 # we have 256 grey level and 5 gradient char. 256/5=51.2
REPRENSATION_GRADIENT=(" ","░","▒","▓","█")
# test datum
class TestData():
    # constructor
    def __init__(self, width=0, height=0, image=[], label="",pattern=""):
        # logger context
        argsStr = methodArgsStringRepresentation(signature(TestData.__init__).parameters,locals())
        # logger input
        logger.loadedLogger.input(__name__, TestData.__name__, TestData.__init__.__name__,message=argsStr)
        # construct object
        self.width = width
        self.height = height
        self.image = image
        self.label = label
        self.pattern = pattern
        # logger output
        logger.loadedLogger.output(__name__, TestData.__name__, TestData.__init__.__name__)
    @staticmethod
    def load(fileName):
        # logger context
        argsStr = methodArgsStringRepresentation(signature(TestData.__init__).parameters, locals())
        # logger input
        logger.loadedLogger.input(__name__, TestData.__name__, TestData.__init__.__name__, message=argsStr)
        # construct object
        file = open(fileName, "rt")
        jsonTestData=file.read()
        file.close()
        testData=ComplexJsonDecoder.loadComplexObject(jsonTestData)
        # logger output
        logger.loadedLogger.output(__name__, TestData.__name__, TestData.__init__.__name__,testData)
        # return
        return testData
    # reprensentation
    def __repr__(self):
        representation=""
        if self.__dict__:
            # string standard data
            standardData=self.__dict__.copy()
            del standardData[IMAGE_MARKUP]
            representation=str(standardData)+linesep
            # string image
            rawIndex=0
            columnIndex=0
            pixelsNumber=self.width*self.height
            for pixelIndex in range(0,pixelsNumber):
                # add pixel related gradient
                pixelValue=self.image[pixelIndex]
                gradientIndex=int(pixelValue/REPRENSATION_INTERVAL)
                gradient=REPRENSATION_GRADIENT[gradientIndex]
                representation=representation+gradient
                # check if new new line
                if columnIndex<self.width-1:
                    columnIndex=columnIndex+1
                else:
                    representation=representation+linesep
                    columnIndex=0
                    rawIndex=rawIndex+1
        return representation
    def __str__(self):
        return self.__repr__()
