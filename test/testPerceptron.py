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
        variables = [-4.759797336992447, 5.224268412298162, -0.8238109165048839, -4.335650860536461, -12.139774266734126, -3.9137186632355467, 2.3090778436678105, -2.6633803360485158, 4.912045083504539, -10.442216563757164]
        expectedValue = [0.008494569592952588, 0.9946445377970348, 0.30495530742883076, 0.012924129215778584, 5.342698960038116e-06, 0.019575273408768638, 0.9096260767034267, 0.0651690933628563, 0.9926963099778543, 2.9173618362790917e-05]
        actualValue = Sigmoid.value(variables)
        actualValue = [float(_) for _ in actualValue]
        self.assertListEqual(expectedValue, actualValue, "ERROR : sigmoïd value does not match")
        pass
    def testSigmoidDerivative(self):
        variables = [0.024861673929273992, 0.28109464441649784, 4.844212871290259e-07, 0.8458432780821669, 0.9974294986897912, 0.5396884426206188, 0.11165856728006232, 0.9982761962289739, 0.9706007225176077, 0.003034795109527507]
        expectedValue = [0.02424357109870845, 0.20208044529686048, 4.844210524650425e-07, 0.130392427005381, 0.0025638938332229657, 0.24842482752234984, 0.09919093163302611, 0.0017208322715850963, 0.02853495996590565, 0.003025585128170695]
        actualValue = Sigmoid.derivative(variables)
        actualValue = [float(_) for _ in actualValue]
        self.assertListEqual(expectedValue, actualValue, "ERROR : sigmoïd value does not match")
        pass
    # test layer computing
    def testDifferentialErrorOutput(self):
        expectedOutput = [0,0,1,0,0,0,0,0,0,0]
        actualOutput = [0.9995057415494378, 0.0005696693299897813, 6.773856462448391e-07, 0.7129941025529047, 0.008378539587771152, 0.0006705591635133233, 0.9652315201365399, 2.07564711785314e-05, 0.984934124954549, 0.0001956411273144735]
        expectedDifferentialError = [0.9995057415494378, 0.0005696693299897813, -0.9999993226143538, 0.7129941025529047, 0.008378539587771152, 0.0006705591635133233, 0.9652315201365399, 2.07564711785314e-05, 0.984934124954549, 0.0001956411273144735]
        actualDifferentialError = Layer.differentialErrorOutput(actualOutput,expectedOutput)
        actualDifferentialError = [float(_) for _ in actualDifferentialError]
        self.assertListEqual(expectedDifferentialError, actualDifferentialError, "ERROR : differential error output does not match")
        pass
    def testDifferentialErrorHidden(self):
        previousDifferentielError = [[0.14183407658948294],[0.14464286433485501],[0.12703726353950687],[0.08500328181266716],[0.09355259185226361],[0.10454788124775924],[0.07756524761984694],[0.033814213071683974],[-0.08176974843303718],[0.0825255766755945]]
        previousLayerWeights = [[-0.7339338396373078,-0.09171190968689302,0.9122745639261545,-0.8513173916984993,-0.8045567236137197,0.3760256831987825,-0.4882869188388077,-0.16430627283202415,-0.1477107907164379,0.6159008535197281,-0.7252812201318208,0.8132571694018083,0.12669455874542357,0.6472288737673857,0.26026542509041617],
            [-0.055150887866475706,-0.8670206367372424,0.0432210688836796,-0.7654661904221913,-0.9519694741261651,0.7522244736756616,0.6025507123354281,0.1853438832690888,-0.0461187754732737,-0.8627466418940204,-0.22009605272489363,0.513250107777429,-0.10882399748643556,0.8558924750865808,0.1640716588563096],
            [0.08807173146658442,0.7782244713793252,-0.9049346091312462,-0.23149227613708634,0.7149975656557102,-0.4999185332254117,0.21375858297494577,0.5866733028777451,0.8761531173193273,0.15397435305961582,-0.9457933332575676,0.39740420778464913,-0.9519379825765892,0.974233426263658,0.6275922245268184],
            [-0.35229939394358234,0.04799057478881452,0.5065128796329623,-0.60469030881464,0.3850504832667181,-0.06417838203463022,-0.8380561619707931,-0.2544993248214522,0.9477148323884432,-0.14902540028474576,0.8919620606403573,0.9223810830236232,-0.2449829632304652,0.9844058272694305,0.908219823522453],
            [-0.05893307437054185,0.5653770226752051,0.3499375315776432,-0.05067925307220578,0.4109552262513123,-0.5700091948879229,0.009649303756931182,-0.6143638270547374,-0.12709619535169003,0.5389043627792118,0.8636496840894341,-0.14288134339112202,0.2812826035570566,-0.5120400688698492,0.11462549144076317],
            [0.1768992855106295,-0.19185808872390253,-0.9236792411331485,-0.5704488327714192,0.49854122580540716,0.5788998407974435,0.9616383384984102,0.6033732403517966,0.6368617502438978,-0.28147250927263334,-0.5262111582015967,0.5360724428034356,-0.5506641586920458,0.591432065606339,0.6448246367795463],
            [-0.7109557836492975,0.5584818622758692,-0.6050828876131416,-0.03178242450170976,-0.383216859999747,-0.24615279149327574,0.5700969741818207,-0.1816139598841513,-0.47423686753531324,0.46998566986474066,-0.7952330413917452,0.5568353416661986,-0.5953101741319509,0.10253030827262233,0.8585439385437519],
            [0.9114487425713862,0.10385893412198066,-0.1538672765111444,0.7458496238727699,0.617931339253667,0.8020821338080604,0.38310696347715334,0.8337150723636286,0.4886782877599145,0.2414930614909696,0.4977770400313333,-0.7897929309964329,-0.5100598380863965,-0.05727040343051226,0.8080436203800161],
            [-0.020349060641251526,0.37483041922364313,-0.5086020357853125,-0.7469501680019115,0.16043699491118746,-0.4729718484291232,0.7074495870118833,0.1848059800017341,-0.8991298935502758,-0.9898446405799668,-0.4852331868857338,-0.3554407266661139,0.20005740942522876,0.027798910604015514,-0.5129507030526008],
            [0.07380317736892184,-0.7727682521787156,0.06536782070630753,0.9021452176115055,0.4744062246665979,-0.6868301984179919,-0.47873278251061335,-0.6365980946218983,0.9018052040566951,0.001548827777451356,0.11302299591561349,-0.11854790648478941,-0.3615901923193121,-0.9353053023398443,0.6587683945732208]]
        expectedDifferentialError = [-0.1344220721562209,-0.05023147223870325,-0.00524659161353101,-0.21837114866269153,-0.020477601224295115,0.09039302155467922,0.03507518596878928,0.008463924322890157,0.3466177496028463,0.11613457899661646,-0.14909963076277233,0.3969203348293972,-0.2804047519432634,0.3635263328280953,0.4859397506679662]
        actualDifferentialError = Layer.differentialErrorHidden(previousDifferentielError,previousLayerWeights)
        actualDifferentialError = [float(_) for _ in actualDifferentialError]
        self.assertListEqual(expectedDifferentialError, actualDifferentialError, "ERROR : differential error hidden does not match")
        pass
    def testcomputeNewWeightsOutput(self):
        layer = Layer()
        layer.weights =
        #input =
        #output =
        #differentialErrorLayer = #input
        #expectedNewDifferentialErrorWeightsBiases =
        #expectedOldWeights =
        #expectedNewWeights =
        #actualNewDifferentialErrorWeightsBiases, actualOldWeights = layer.computeNewWeights(input,output,differentialErrorLayer)
        #self.assertListEqual(expectedNewDifferentialErrorWeightsBiases, actualNewDifferentialErrorWeightsBiases, "ERROR : NewDifferentialErrorWeightsBiases does not match")
        #self.assertListEqual(expectedOldWeights, actualOldWeights, "ERROR : OldWeights does not match")
        #self.assertListEqual(expectedNewWeights, layer.weights, "ERROR : NewWeights does not match")
        pass
    def testcomputeNewWeightsHidden(self):
        input = [0.9850495139393386,0.8990937319023146,0.4894580292431677,0.04424514894552958,0.0833369232101113,0.01309118385415899,0.04260004058804179,0.734036287580626,0.0918593281080508,0.44201051512685424,0.8750156839103664,0.7623921853990762,0.8916009110719337,0.9715872191936142,0.8821322604458753]
        output = []
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
