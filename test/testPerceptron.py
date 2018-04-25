#!/usr/bin/env python3
# coding=utf-8
# import
from unittest import TestCase
from random import randint, choice
from string import ascii_letters
from neuralnetworkcommon.perceptron import Perceptron, Layer, Sigmoid
# test perceptron
class testPerceptron(TestCase):
    # test sigmoid computing
    def testSigmoidValue(self):
        # random perceptron
        variables = [-4.759797336992447, 5.224268412298162, -0.8238109165048839, -4.335650860536461, -12.139774266734126, -3.9137186632355467, 2.3090778436678105, -2.6633803360485158, 4.912045083504539, -10.442216563757164]
        expectedValue = [0.008494569592952588, 0.9946445377970348, 0.30495530742883076, 0.012924129215778584, 5.342698960038116e-06, 0.019575273408768638, 0.9096260767034267, 0.0651690933628563, 0.9926963099778543, 2.9173618362790917e-05]
        actualValue = Sigmoid.value(variables)
        actualValue = [float(_) for _ in actualValue]
        self.assertListEqual(expectedValue, actualValue, "ERROR : sigmoïd value does not match")
        pass
    def testSigmoidDerivative(self):
        # random perceptron
        variables = [0.024861673929273992, 0.28109464441649784, 4.844212871290259e-07, 0.8458432780821669, 0.9974294986897912, 0.5396884426206188, 0.11165856728006232, 0.9982761962289739, 0.9706007225176077, 0.003034795109527507]
        expectedValue = [0.02424357109870845, 0.20208044529686048, 4.844210524650425e-07, 0.130392427005381, 0.0025638938332229657, 0.24842482752234984, 0.09919093163302611, 0.0017208322715850963, 0.02853495996590565, 0.003025585128170695]
        actualValue = Sigmoid.derivative(variables)
        actualValue = [float(_) for _ in actualValue]
        self.assertListEqual(expectedValue, actualValue, "ERROR : sigmoïd value does not match")
        pass
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
