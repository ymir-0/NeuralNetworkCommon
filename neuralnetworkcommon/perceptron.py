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
    pass
pass
