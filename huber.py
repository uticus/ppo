#!/usr/bin/env python

#import agents

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

class HuberLoss(tf.keras.losses.Loss):

    def __init__(self,
                 clip_value: float,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='HuberLoss'):

        self.clip_value = clip_value
        self.isinf_ = np.isinf(clip_value)
        super().__init__(reduction=reduction, name=name)

    def _loss(self, y_true, y_pred):
        assert self.clip_value > 0.

        x = y_true - y_pred

        squared_loss = 0.5 * K.square(x)

        # nothing but check
        if self.isinf_:
            return K.mean(squared_loss)

        condition = K.abs(x) < self.clip_value
        linear_loss = self.clip_value * (K.abs(x) - 0.5 * self.clip_value)
        loss = tf.where(condition, squared_loss, linear_loss)  # condition, true, false
        #return K.mean(loss, axis=-1)
        return K.mean(loss)

    def call(self, y_true, y_pred):
        return self._loss(y_true=y_true, y_pred=y_pred)


# huber_loss_function(np.inf) is equal to mean_squared_loss

class MeanSquaredLoss(tf.keras.losses.Loss):
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='MeanSquaredLoss'):
        super().__init__(reduction=reduction, name=name)

    def _loss(self, y_true, y_pred):
        x = y_true - y_pred
        squared_loss = 0.5 * K.square(x)
        return K.mean(squared_loss)
    #return K.mean(0.5 * K.square(y_true - y_pred) )

    def call(self, y_true, y_pred):
        return self._loss(y_true=y_true, y_pred=y_pred)




def test():


    shape = (6, 7)
    y_a = np.random.random(shape)
    y_b = np.random.random(shape)

    loss = HuberLoss(clip_value=1.0)

    out1 = K.eval(loss(K.variable(y_a), K.variable(y_b)))
    out2 = K.eval( loss(y_a, y_b) )

    print(out1)
    print(out2)
    pass



if __name__ == "__main__":
    
    test()

    print("passed")
    input("Enter to exit")
    exit()
