# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:23:43 2022

@author: Tomasz Konieczka
"""

import numpy as np

class auxiliary:
    @staticmethod
    def dataNoise(data, sigma):
        x_train = data
        noise = np.random.normal(0., sigma, (x_train.shape))
        x_train += noise
        return x_train
    
    @staticmethod
    def dataCentering(data, center=0.0):
        return (data - np.mean(data, axis=0) + center)

    @staticmethod
    def dataNorm(data):
        return ( (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)) )
                 
    @staticmethod
    def dataStandardization(data):
        return ( auxiliary.dataCentering(data) / np.std(data, axis=0) )
    
    @staticmethod
    def dataDividing(data):
        evaluateData = data
        half_of_data = int(len(evaluateData)/2)
        (x_train, y_train), (x_test, y_test) = (evaluateData[:half_of_data, 0], evaluateData[:half_of_data, 1]), (
            evaluateData[(half_of_data):, 0], evaluateData[(half_of_data):, 1])
        return (x_train, y_train), (x_test, y_test)