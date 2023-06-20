# -*- coding: utf-8 -*-
"""
Created on 

@author: Tomasz Konieczka
"""

import tensorflow as tf
import numpy as np
import dane

class tf_test:
    
    def __init__(self):
        pass

    def tf_test(self, data_set):
        (x_train, y_train), (x_test, y_test) = dane.auxiliary.dataDividing(data_set)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(1),
            # 128 -- ten parametr najwiecej wplywa na optymalizacje, mozna jeszcze zmieniac fun aktywacji (zd8)
            tf.keras.layers.Dense(10, activation='tanh'),
            tf.keras.layers.Dense(1, activation='ReLU')
        ])

        # tf.nn.softmax(predictions).numpy()
        loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
        self.model.compile(optimizer='adagrad',
                      loss=loss_fn,
                      metrics=['Poisson'])

        self.model.fit(tf.cast(x_train, tf.float32),
                  tf.cast(y_train, tf.float32), epochs=5)
        self.model.evaluate(tf.cast(x_test, tf.float32),
                       tf.cast(y_test, tf.float32), verbose=2)
        self.weights0 = self.model.layers[0].get_weights()[0]
        self.bias0 = self.model.layers[0].bias.numpy()
        self.weights1 = self.model.layers[1].get_weights()[0]
        self.bias1 = self.model.layers[1].bias.numpy()
       # print(self.model.layers[0].bias.numpy())

    



  