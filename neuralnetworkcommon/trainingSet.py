# coding=utf-8
# import
from neuralnetworkcommon.trainingElement import TrainingElement, separateData
from pythoncommontools.objectUtil.objectUtil import Bean
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
        return separateData(self.trainingElements)
    pass
pass
