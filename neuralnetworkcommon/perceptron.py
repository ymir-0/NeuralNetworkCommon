# coding=utf-8
# import
from numpy.random import rand
# layer
class Layer():
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
    pass
# perceptron
class Perceptron():
    def __init__(self,dimensions,comments=None):
        # create each layer
        self.layers = [Layer(dimensions[index],dimensions[index+1]) for index in range(len(dimensions)-1)]
        # save comments
        self.comments = comments
        pass
    pass
pass
