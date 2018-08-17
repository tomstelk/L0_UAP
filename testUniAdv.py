# -*- coding: utf-8 -*-
"""
Created on Wed May 23 19:07:05 2018

@author: user
"""

import tensorflow as tf

def testUniAdv(model, x_test, y_test, perb, perb_target,clip_min,clip_max):
    
    sizeTestSample = tf.cast(tf.shape(y_test)[0],tf.float32)
    
    #Original accuracy
    preds = tf.argmax(model.get_probs(x_test), axis = 1)
    acc = tf.reduce_sum(tf.cast(tf.equal(preds, tf.argmax(y_test, axis=1)),tf.float32)) \
        / sizeTestSample
    
    #Accuracy with perb
    x_perb = tf.clip_by_value(x_test+perb, clip_min,clip_max)
    predsWithPerb = tf.argmax(model.get_probs(x_perb), axis = 1)
    accWithPerb = tf.reduce_sum(tf.cast(tf.equal(predsWithPerb, tf.argmax(y_test, axis=1)),tf.float32)) \
        / sizeTestSample
    
    #Perb accuracy 
    perbAcc = tf.reduce_sum(tf.cast(tf.equal(predsWithPerb, tf.argmax(perb_target, axis=1)),tf.float32)) \
        / sizeTestSample
    
    return acc,accWithPerb,perbAcc
    