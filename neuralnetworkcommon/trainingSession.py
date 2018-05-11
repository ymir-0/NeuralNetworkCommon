# coding=utf-8
# import
from pythoncommontools.objectUtil.objectUtil import Bean
from random import shuffle
from neuralnetworkcommon.trainingElement import separateData
# training session
class TrainingSession(Bean):
    # constructors
    # INFO : this dummy constructor is requested for complex JSON (en/de)coding
    def __init__(self):
        self.perceptronId = 0
        self.trainingSessionId = 0
    @staticmethod
    # INFO : test ration between 0 (no data used to test, all for training) and 1 (no data used to training, all for test)
    def constructFromTrainingSet(perceptronId,trainingSet,testRatio,comments=""):
        # initialize training set
        trainingSession = TrainingSession()
        # add ids & comments
        trainingSession.perceptronId = perceptronId
        trainingSession.trainingSessionId = trainingSet.id
        trainingSession.comments = comments
        # split training / test sets
        dataElements = trainingSet.trainingElements
        shuffle(dataElements)
        testSetLength = int(len(dataElements)*testRatio)
        trainingSession.testSet = dataElements[:testSetLength]
        trainingSession.trainingSet = dataElements[testSetLength:]
        # return
        return trainingSession
    @staticmethod
    def constructFromAttributes(perceptronId,trainingSessionId,trainingSet,testSet,status,pid,meanDifferantialErrors,trainedElementsNumbers,errorElementsNumbers,comments=""):
        # initialize training set
        trainingSession = TrainingSession()
        # add attributs
        trainingSession.perceptronId = perceptronId
        trainingSession.trainingSessionId = trainingSessionId
        trainingSession.trainingSet = trainingSet
        trainingSession.testSet = testSet
        trainingSession.status = status
        trainingSession.pid = pid
        trainingSession.meanDifferantialErrors = meanDifferantialErrors
        trainingSession.trainedElementsNumbers = trainedElementsNumbers
        trainingSession.errorElementsNumbers = errorElementsNumbers
        trainingSession.comments = comments
        # return
        return trainingSession
    # separate data
    def separateData(self):
        trainingInputs,trainingExpectedOutputs = separateData(self.trainingSet)
        testInputs,testExpectedOutputs = separateData(self.testSet)
        return trainingInputs,trainingExpectedOutputs, testInputs,testExpectedOutputs
    pass
pass