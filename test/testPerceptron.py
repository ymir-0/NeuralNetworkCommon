#!/usr/bin/env python3
# coding=utf-8
# import
from unittest import TestCase
from random import randint, choice
from string import ascii_letters
from neuralnetworkcommon.perceptron import Perceptron
# test perceptron
class testPerceptron(TestCase):
    # test constructor
    def testConstructor(self):
        # randomize layers numbers, dimensions & comments
        layersNumber = randint(2,12)
        dimensions = [randint(2,100) for _ in range(layersNumber)]
        comments = "".join([choice(ascii_letters) for _ in range(15)])
        # construct perceptron
        perceptron = Perceptron(dimensions,comments)
        # check layers dimensions
        differentWeights = False
        perceptronLayersDimension = layersNumber-1
        self.assertEqual(perceptronLayersDimension, len(perceptron.layers), "ERROR : perceptron layers number")
        self.assertEqual(dimensions[0], perceptron.layers[0].weights.shape[1], "ERROR : first layer column dimension")
        self.assertEqual(dimensions[-1], perceptron.layers[-1].weights.shape[0], "ERROR : last layer row dimension")
        for layerIndex in range(perceptronLayersDimension):
            firstWeight = perceptron.layers[layerIndex].weights[0][0]
            if layerIndex < perceptronLayersDimension-1 :
                nextLayerIndex = layerIndex+1
                self.assertEqual(dimensions[nextLayerIndex], perceptron.layers[nextLayerIndex].weights.shape[1], "ERROR : next layer column dimensions")
                self.assertEqual(perceptron.layers[layerIndex].weights.shape[0], perceptron.layers[nextLayerIndex].weights.shape[1], "ERROR : layers row/column dimensions")
            rowsNumber = perceptron.layers[layerIndex].weights.shape[0]
            self.assertEqual(perceptron.layers[layerIndex].biases, [0]*rowsNumber, "ERROR : bias != 0")
            for row in range(rowsNumber):
                for column in range(perceptron.layers[layerIndex].weights.shape[1]):
                    currentWeight = perceptron.layers[layerIndex].weights[row][column]
                    self.assertGreaterEqual(currentWeight, -1, "ERROR : weight < -1")
                    self.assertLessEqual(currentWeight, 1, "ERROR : weight > 1")
                    if not differentWeights : differentWeights = firstWeight != currentWeight
                    pass
                pass
            self.assertTrue(differentWeights, "ERROR : all weights are the same")
            pass
        # check comments
        self.assertEqual(comments, perceptron.comments, "ERROR : perceptron comment")
        pass
    pass
pass
