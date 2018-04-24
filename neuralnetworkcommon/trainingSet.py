# coding=utf-8
# import
from pythoncommontools.objectUtil.objectUtil import Bean
from random import random
# training element
def mergeData(inputs, expectedOutputs):
    trainingElements = list()
    for index, input in enumerate(inputs):
        expectedOutput = expectedOutputs[index]
        trainingElement = TrainingElement.constructFromAttributes(input,expectedOutput)
        trainingElements.append(trainingElement)
    return trainingElements
class TrainingElement(Bean):
    # constructors
    # INFO : this dummy constructor is requested for complex JSON (en/de)coding
    def __init__(self):
        self.input = []
        self.expectedOutput = []
    @staticmethod
    def constructFromAttributes(input,expectedOutput):
        # initialize training element
        trainingElement = TrainingElement()
        # add attributs
        trainingElement.input=input
        trainingElement.expectedOutput=expectedOutput
        # return
        return trainingElement
    # JSON marshall / unmarshall
    def jsonMarshall(self):
        jsonTrainingElement = dict(self.__dict__)
        return jsonTrainingElement
    @staticmethod
    def jsonUnmarshall(**attributes):
        trainingElement = TrainingElement()
        trainingElement.__dict__.update(attributes)
        return trainingElement
    pass
# training set
class TrainingSet(Bean):
    # constructors
    # INFO : this dummy constructor is requested for complex JSON (en/de)coding
    def __init__(self):
        # INFO : technically, a set must contains hashable elements only (list is not one of them). So we use a list to embed lists
        self.trainingElements = list()
        self.comments = ""
    @staticmethod
    def constructFromAttributes(id,trainingElements,comments=""):
        # initialize training set
        trainingSet = TrainingSet()
        # add attributs
        trainingSet.id=id
        trainingSet.trainingElements=trainingElements
        trainingSet.comments=comments
        # return
        return trainingSet
    # JSON marshall / unmarshall
    def jsonMarshall(self):
        # marshall perceptron
        jsonTrainingSet = dict(self.__dict__)
        # marshall each training element
        trainingElements=list()
        for trainingElement in jsonTrainingSet["trainingElements"]:
            trainingElements.add(trainingElement.jsonMarshall())
            jsonTrainingSet["trainingElements"] = trainingElements
        # return
        return jsonTrainingSet
    @staticmethod
    def jsonUnmarshall(**attributes):
        # initialize training set
        trainingSet = TrainingSet()
        # unmarshall perceptron
        trainingSet.__dict__.update(attributes)
        # unmarshall each layer
        for trainingElementIndex, jsonTrainingElement in enumerate(trainingSet.trainingElements):
            trainingSet.trainingElements[trainingElementIndex] = TrainingElement.jsonUnmarshall(**jsonTrainingElement)
        # return
        return trainingSet
    # separate data
    def separateData(self):
        inputs = list()
        expectedOutputs = list()
        for trainingElement in self.trainingElements:
            inputs.append(trainingElement.input)
            expectedOutputs.append(trainingElement.expectedOutput)
        return inputs, expectedOutputs
    pass
pass
