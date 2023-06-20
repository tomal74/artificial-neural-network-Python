# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:12:22 2022

*funkcje aktywacji dla ssn

@author: Tomasz Konieczka
"""

import numpy as np


class activation:   
    @staticmethod
    def none(u, deriv=False):
        if deriv==False:
            return u
        else:
            u=np.where(u==u,1,1)
            return u
        
    @staticmethod
    def ReLU(u, deriv=False):
        if deriv==False:
            return np.maximum(0, u)
        else:
            u=np.where(u>=0,1,u)
            return u

    @staticmethod
    def tanh(u, deriv=False):
        if deriv==False:
            return np.tanh(u)
        else:
            return (1-u**2)

    @staticmethod
    def sigmoid(u, deriv=False):
        if deriv==False:
            return (1/(1 + np.exp(-u)))
        else:
            return (u*(1-u))

    @staticmethod
    def ELU(u, alpha=1.0, deriv=False):
        if deriv==False:
            return np.where(u>=0, u, alpha*(np.exp(u)-1))
        else:
            return np.where(u<0, u+alpha, 1)
            
     

