# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:43:08 2022

@author: Tomasz Konieczka 

warstwa.py
"""
import numpy as np


class warstwa:

    def __init__(self, neurons_no, activation_function):
        if(int(neurons_no) in range(1, int(1e7))):
            self.neurons_no = int(neurons_no)   # liczba neuronow w warstwie
        else:
            raise Exception(
                'ERR! - podana liczba neuronow jest nieprawidlowa - ({}) - liczba neuronow musi byc pomiedzy 1 a 1e7'.format(int(neurons_no)))
        self.activation_fun = activation_function  # funkcja aktywacji
        self.W = None    # wektor wag
        self.bias = 0

    def __multip(self, W, x):  # mnozenie W*x
        return np.matmul(W, x)

    def layer(self, W, x):
        self.__sizeCheck(W, x)
        self.x = x
        out_sum = self.__multip(W, x)
        out_sum += self.bias
        #print(out_sum)
        self.out = self.activation_fun(out_sum)
        return self.out

    def __sizeCheck(self, W, x):
        if(W.shape[1] != x.shape[0]):
            raise Exception("ERR - wymiar wag nie zgadza sie z wymiarem wejsc")

    def back_prop(self, loss):
        self.act_prim = loss * self.activation_fun(self.out, deriv=True)
        self.w_prim = np.matmul(self.act_prim, np.transpose(self.x))
        self.x_prim = np.matmul(np.transpose(self.W), self.act_prim)
        return self.act_prim, self.w_prim, self.x_prim

