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
        perceptronLayersDimension = layersNumber-1
        self.assertEqual(perceptronLayersDimension, len(perceptron.layers), "ERROR : perceptron layers number")
        self.assertEqual(dimensions[0], len(perceptron.layers[0].weights[1]), "ERROR : first layer column dimension")
        for layerIndex in range(perceptronLayersDimension):
            if layerIndex < perceptronLayersDimension-1 :
                nextLayerIndex = layerIndex+1
                self.assertEqual(dimensions[nextLayerIndex], perceptron.layers[nextLayerIndex].weights.shape[1], "ERROR : next layer column dimensions")
                self.assertEqual(perceptron.layers[layerIndex].weights.shape[0], perceptron.layers[nextLayerIndex].weights.shape[1], "ERROR : layers row/column dimensions")
            for row in range(perceptron.layers[layerIndex].weights.shape[0]):
                self.assertEqual(perceptron.layers[layerIndex].biases[row], 0, "ERROR : bias != 0")
                for column in range(perceptron.layers[layerIndex].weights.shape[1]):
                    self.assertGreaterEqual(perceptron.layers[layerIndex].weights[row][column], -1, "ERROR : weight < -1")
                    self.assertLessEqual(perceptron.layers[layerIndex].weights[row][column], 1, "ERROR : weight > 1")
                    pass
                pass
            pass
        # check comments
        self.assertEqual(comments, perceptron.comments, "ERROR : perceptron comment")
        pass
    pass
pass
