#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Spyder Editor
Tomasz Konieczka 
This is a temporary script file.
"""

# run  --> python3 projekt.py plik.json dane.txt

import sys
import numpy as np
from funkcje import activation
from dane import auxiliary
from siec import siec as model
from warstwa import warstwa
from tf import tf_test


if __name__ == "__main__":
    #print(sys.argv)
    print('Number of arguments:', len(sys.argv)-1, 'arguments.\n')
    if(len(sys.argv) > 3):
        print('ERR!! - Zbyt duzo parametrow wejsciowych.\nPodaj dwa parametry...')
        sys.exit()
    
    data_raw = np.genfromtxt(sys.argv[2] , delimiter=' ', usecols=(1, 2))
    (x_train, y_train), (x_test, y_test) = auxiliary.dataDividing(data_raw)   
    
    tf_model = tf_test()
    tf_model.tf_test(data_raw)
    
    mod = model( [warstwa(3, activation.tanh)], sys.argv[1])
    mod.layers[1].W = np.transpose(tf_model.weights0)
    mod.layers[2].W = np.transpose(tf_model.weights1)
    
    mod.layers[1].bias = np.transpose(tf_model.bias0)
    mod.layers[2].bias = np.transpose(tf_model.bias1)
    pred = mod(x_test[100:101])
    print(y_test[100:101])
    
    print('\n\n  zaszumienie danych w pliku dane.py!')
    print('\n dwie przykladowe dane treningowe przed zaszumieniem: {}'.format(x_train[8:10]))
    #zaszumienie danych treningowych
    x_train = auxiliary.dataNoise(x_train, 5)
    print('\n dwie przykladowe dane treningowe po zaszumieniu: {}'.format(x_train[8:10]))
    
    print('\n\nodpowiedz wlasnej sieci: ')
    print(*pred)
    print('\nodpowiedz tf')
    print(tf_model.model(x_test[100:101]))
