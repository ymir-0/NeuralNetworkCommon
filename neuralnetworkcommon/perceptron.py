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
        # initialize object
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
        # initialize object
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
        for layerIndex, layerObject in enumerate(jsonPerceptron["layers"]):
            layers.append(layerObject.jsonMarshall())
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
        self.comments = ""
    @staticmethod
    def constructFromAttributes(id,input,expectedOutput,comments=""):
        # initialize object
        trainingElement = TrainingElement()
        # add attributs
        trainingElement.id=id
        trainingElement.input=input
        trainingElement.expectedOutput=expectedOutput
        trainingElement.comments=comments
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
pass
