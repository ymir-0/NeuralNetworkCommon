#!/usr/bin/env python3
# coding=utf-8
# import
from unittest import TestCase
from random import randint, choice
from string import ascii_letters
from neuralnetworkcommon.perceptron import Perceptron, Layer
# test perceptron
class testPerceptron(TestCase):
    # test constructor
    @staticmethod
    def getRandomPerceptron():
        # randomize layers numbers, dimensions & comments
        layersNumber = randint(2,12)
        dimensions = [randint(2,100) for _ in range(layersNumber)]
        comments = "".join([choice(ascii_letters) for _ in range(15)])
        # construct perceptron
        perceptron = Perceptron.constructRandomFromDimensions(dimensions,comments)
        # return
        return perceptron, layersNumber, dimensions, comments
    def testDefaultConstructor(self):
        # random perceptron
        perceptron, layersNumber, dimensions, comments = testPerceptron.getRandomPerceptron()
        # check layers dimensions
        differentWeights = False
        perceptronLayersDimension = layersNumber-1
        self.assertEqual(perceptronLayersDimension, len(perceptron.layers), "ERROR : perceptron layers number")
        self.assertEqual(dimensions[0], len(perceptron.layers[0].weights[0]), "ERROR : first layer column dimension")
        self.assertEqual(dimensions[-1], len(perceptron.layers[-1].weights), "ERROR : last layer row dimension")
        for layerIndex in range(perceptronLayersDimension):
            firstWeight = perceptron.layers[layerIndex].weights[0][0]
            if layerIndex < perceptronLayersDimension-1 :
                nextLayerIndex = layerIndex+1
                self.assertEqual(dimensions[nextLayerIndex], len(perceptron.layers[nextLayerIndex].weights[0]), "ERROR : next layer column dimensions")
                self.assertEqual(len(perceptron.layers[layerIndex].weights), len(perceptron.layers[nextLayerIndex].weights[0]), "ERROR : layers row/column dimensions")
            rowsNumber = len(perceptron.layers[layerIndex].weights)
            self.assertEqual(perceptron.layers[layerIndex].biases, [0]*rowsNumber, "ERROR : bias != 0")
            for row in range(rowsNumber):
                for column in range(len(perceptron.layers[layerIndex].weights[0])):
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
    # test constructor
    def testPerceptronConstructorFromAttributes(self):
        # random perceptron
        initialPerceptron, _, _, _ = testPerceptron.getRandomPerceptron()
        initialPerceptron.id = randint(0, 1000)
        # construct from attributes
        constructedPerceptron = Perceptron.constructFromAttributes(initialPerceptron.id,initialPerceptron.layers,initialPerceptron.comments)
        # check construction
        self.assertTrue(initialPerceptron==constructedPerceptron, "ERROR : perceptron not consistent with attributs")
        pass
    def testLayerConstructorFromAttributes(self):
        # random layer
        previousDimension = randint(2,12)
        currentDimension = randint(2,12)
        initialLayer=Layer.constructRandomFromDimensions(previousDimension, currentDimension)
        # construct from attributes
        constructedLayer = Layer.constructFromAttributes(initialLayer.weights,initialLayer.biases)
        # check construction
        self.assertTrue(initialLayer==constructedLayer, "ERROR : layer not consistent with attributs")
        pass
    pass
pass
