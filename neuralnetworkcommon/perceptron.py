# coding=utf-8
# import
from numpy.random import rand
from pythoncommontools.objectUtil.objectUtil import Bean
# layer
class Layer(Bean):
    # constructors
    '''
    TODO : add extras parameters (uncertainties/dilatations/offsets)
    all parameters should be randomized between some given ranges
    TODO : parallelize random array generation
    '''
    def __init__(self,previousDimension, currentDimension):
        # random weights between -1/+1
        self.weights = (rand(currentDimension, previousDimension) - .5) * 2
        # set biases to 0
        self.biases = [0] * currentDimension
        pass
    @staticmethod
    def constructFromAttributes(weights=None,biases=None):
        # initialize object
        layer = Layer(0,0)
        # add attributs
        layer.weights=weights
        layer.biases=biases
        # return
        return layer
    pass
# perceptron
class Perceptron(Bean):
    # constructors
    def __init__(self,dimensions,comments=None):
        # create each layer
        self.layers = [Layer(dimensions[index],dimensions[index+1]) for index in range(len(dimensions)-1)]
        # save comments
        self.comments = comments
        pass
    @staticmethod
    def constructFromAttributes(id,layers=None,comments=None):
        # initialize object
        perceptron = Perceptron([])
        # add attributs
        perceptron.id=id
        perceptron.layers=layers
        perceptron.comments=comments
        # return
        return perceptron
    pass
pass
