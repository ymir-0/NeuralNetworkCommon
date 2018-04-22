# coding=utf-8
# import
from pythoncommontools.objectUtil.objectUtil import Bean
from random import random
# layer
class Layer(Bean):
    # constructors
    '''
    TODO : add extras parameters (uncertainties/dilatations/offsets)
    all parameters should be randomized between some given ranges
    TODO : parallelize random array generation
    '''
    '''
    INFO : numpy arrays are not standrad object.
    they can not  transnsform attributs as map so they can not be easily transformed in JSON
    so we use a standard array for construction and will transforme it in numpy later
    '''
    # INFO : this dummy constructor is requested for complex JSON (en/de)coding
    def __init__(self):
        self.weights = [[]]
        self.biases = []
    @staticmethod
    def constructRandomFromDimensions(previousDimension, currentDimension):
        # initialize object
        layer = Layer()
        # random weights between -1/+1
        layer.weights = [[(random() - .5) * 2 for _ in range(previousDimension)] for _ in range(currentDimension)]
        # set biases to 0
        layer.biases = [0] * currentDimension
        # return
        return layer
    @staticmethod
    def constructFromAttributes(weights,biases):
        # initialize layer
        layer = Layer()
        # add attributs
        layer.weights=weights
        layer.biases=biases
        # return
        return layer
    # JSON marshall / unmarshall
    def jsonMarshall(self):
        jsonLayer = dict(self.__dict__)
        return jsonLayer
    @staticmethod
    def jsonUnmarshall(**attributes):
        # initialize layer
        layer = Layer()
        # unmarshall perceptron
        layer.__dict__.update(attributes)
        # return
        return layer
    pass
# perceptron
class Perceptron(Bean):
    # constructors
    # INFO : this dummy constructor is requested for complex JSON (en/de)coding
    def __init__(self):
        self.layers = []
        self.comments = ""
    @staticmethod
    def constructRandomFromDimensions(dimensions,comments=""):
        # initialize object
        perceptron = Perceptron()
        # create each layer
        perceptron.layers = [Layer.constructRandomFromDimensions(dimensions[index],dimensions[index+1]) for index in range(len(dimensions)-1)]
        # save comments
        perceptron.comments = comments
        # return
        return perceptron
    @staticmethod
    def constructFromAttributes(id,layers,comments=""):
        # initialize perceptron
        perceptron = Perceptron()
        # add attributs
        perceptron.id=id
        perceptron.layers=layers
        perceptron.comments=comments
        # return
        return perceptron
    # JSON marshall / unmarshall
    def jsonMarshall(self):
        # marshall perceptron
        jsonPerceptron = dict(self.__dict__)
        # marshall each layer
        layers=list()
        for layer in jsonPerceptron["layers"]:
            layers.append(layer.jsonMarshall())
        jsonPerceptron["layers"] = layers
        # return
        return jsonPerceptron
    @staticmethod
    def jsonUnmarshall(**attributes):
        # initialize perceptron
        perceptron = Perceptron()
        # unmarshall perceptron
        perceptron.__dict__.update(attributes)
        # unmarshall each layer
        for layerIndex, jsonLayer in enumerate(perceptron.layers):
            perceptron.layers[layerIndex] = Layer.jsonUnmarshall(**jsonLayer)
        # return
        return perceptron
    pass
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
