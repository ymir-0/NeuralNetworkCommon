# coding=utf-8
# import
from pythoncommontools.objectUtil.objectUtil import Bean
# utilities
def separateData(trainingElements):
    inputs = list()
    expectedOutputs = list()
    for trainingElement in trainingElements:
        inputs.append(trainingElement.input)
        expectedOutputs.append(trainingElement.expectedOutput)
    return inputs, expectedOutputs
def mergeData(inputs, expectedOutputs):
    trainingElements = list()
    for index, input in enumerate(inputs):
        expectedOutput = expectedOutputs[index]
        trainingElement = TrainingElement.constructFromAttributes(input,expectedOutput)
        trainingElements.append(trainingElement)
    return trainingElements
# training element
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
