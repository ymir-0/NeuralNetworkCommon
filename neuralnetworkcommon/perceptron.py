# coding=utf-8
# import
from numpy import exp, array
from pythoncommontools.objectUtil.objectUtil import Bean
from random import random
# sigmoid
# TODO : create an abstract class for all future functions
# TODO : compute with spark each method
# TODO : add extra parameters : uncertainties, dilatation, offsets
class Sigmoid():
    @staticmethod
    def value(variables):
        arrayVariables = array(variables)
        #value = dilatations / (1 + exp(-array(arrayVariables) * uncertainties)) + offsets
        value = 1 / (1 + exp(-arrayVariables))
        return value
    @staticmethod
    # INFO : we compute the derivative from : value = sigmoïd(variables)
    def derivative(variables):
        arrayVariables = array(variables)
        #derivative = dilatations * uncertainties * arrayVariables * (1 - arrayVariables)
        derivative = variables * (1 - arrayVariables)
        return derivative
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
    # get differential error on output layer
    @staticmethod
    def differentialErrorOutput(actualOutput,expectedOutput):
        # TODO : compute with spark 'differentialError'
        differentialError = array(actualOutput) - array(expectedOutput)
        return differentialError
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
pass
