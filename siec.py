# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:51:59 2022

@author: G505s
"""

from warstwa import warstwa 
import numpy as np
import json
from funkcje import activation


class siec:
    """konstruktor inicjalizuje model na dwa sposoby - w sposob podstawowy, kiedy jawnie podajemy
    do argumentu @layers warstwy lub w drugi sposob - jezeli podamy plik json to argument @layers
    zostanie pominiety a warstwy sieci beda zgodne z plikiem .json"""

    def __init__(self, layers, json_file=None):
        if(json_file != None):
            self.layers = self.init_from_json(json_file)
        else:
            self.layers = layers

        for i in range(len(self.layers)):
            if(i == 0):  # zakladamy - zgodnie z pierwszym wykladem, ze liczba wejsc w 1. warstwie jest rowna liczbie neuronow w tej warstwie
                self.layers[i].W = np.eye(
                    self.layers[i].neurons_no)
            else:   # w dalszych warstwach przyznajemy liczbe wejsc na podstawie l.neu i liczby wyj. z poprzedniej warstwy
                self.layers[i].W = np.random.rand(
                    self.layers[i].neurons_no, self.layers[i-1].neurons_no)

    def __call__(self, x):
        for i in range(len(self.layers)):
            if(i == 0):
                self.layers[i].layer(self.layers[i].W, x)
            else:
                self.layers[i].layer(self.layers[i].W, self.layers[i-1].out)
        # wyjscie z sieci = wyjscie z ostatniej warstwy
        return self.layers[-1].out

    def init_from_json(self, json_file):
        layer_json_init = open(json_file, "r", encoding='utf-8')
        try:
            json_init = json.load(layer_json_init)
        finally:
            layer_json_init.close()

        layers_acv_fun = []
        layers_neu_no = []
        self.model_info = []
        for i in json_init['warstwy']:
           temp_acv = i['funkcja_aktywacji']
           temp_no = i['liczba_neuronów']
           layers_acv_fun.append(temp_acv)
           layers_neu_no.append(temp_no)
           temp_info = '{:<5}-> {:>3}'
           self.model_info.append(temp_info.format(str(int(temp_no)), temp_acv))

        acv_fun_translator = {'ELU': activation.ELU, 'ReLU': activation.ReLU,
                              'sig': activation.sigmoid, 'tanh': activation.tanh, 'none': activation.none}
        layers_array = []
        for i in range(len(layers_acv_fun)):
            layers_array.append(
                warstwa(layers_neu_no[i], acv_fun_translator[layers_acv_fun[i]]))
        return layers_array
