
import tensorflow as tf
from cleverhans.utils_mnist import data_mnist
from cleverhans_tutorials.tutorial_models import make_basic_cnn
import numpy as np
# Get MNIST test data


def convertOneHot2Labels(oneHotLabels):
    return [np.where(r==1)[0][0] for r in oneHotLabels]
'''
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
    advRate = tf.reduce_sum(tf.cast(tf.equal(predsWithPerb, tf.argmax(perb_target, axis=1)),tf.float32)) \
        / sizeTestSample
    
    return acc,accWithPerb,advRate

'''
    
def getModelAcc_Symbolic(model, x_test, y_test):
    sizeTestSample = tf.cast(tf.shape(y_test)[0],tf.float32)
    preds = tf.argmax(model.get_probs(x_test), axis = 1)
    acc = tf.reduce_sum(tf.cast(tf.equal(preds, tf.argmax(y_test, axis=1)),tf.float32)) \
        / sizeTestSample
    return acc

def getAdvRate_Symbolic(orig_preds, perb_preds, perb_target):
    sizeTestSample = sizeTestSample = tf.cast(tf.shape(orig_preds)[0],tf.float32)
    advRate = tf.reduce_sum(tf.cast(tf.equal(perb_preds, tf.argmax(perb_target, axis=1)),tf.float32)) \
        / sizeTestSample
        
    return advRate
    

def filterSampleSet(y_in, target, orig_preds):
    y_labels=convertOneHot2Labels(y_in)
    rep_t = np.tile(target, len(y_labels))
    idx = [all([y == o, y != t]) for (y,o,t) in zip(y_labels, orig_preds, rep_t)]
    return idx

def getModelPreds(x_in, modelFunc, modelFile):
    in_shape=x_in.shape
    x=tf.placeholder(dtype= tf.float32,shape=(None,in_shape[1],in_shape[2],in_shape[3]))
    model=modelFunc()
    model_preds = tf.argmax(model.get_probs(x), axis = 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, modelFile)
        model_preds_np =sess.run(model_preds,feed_dict={x:x_in})
    
    return model_preds_np

    
''' 
train_start=0
train_end=60000
test_start=0
test_end=10000
X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
   
tf.reset_default_graph()
cnn=make_basic_cnn()
x=tf.placeholder(dtype = tf.float32, shape=(None, 28,28,1))
orig_preds = tf.argmax(cnn.get_probs(x), axis = 1)


saver = tf.train.Saver()
cnnModelFile = "../UAN/savedmodels/basic_cnn_MNIST/basic_cnn.ckpt"    



with tf.Session() as sess:
    saver.restore(sess, cnnModelFile)
    orig_preds_np = sess.run(orig_preds, feed_dict={x:X_test})

y_test_labels = convertOneHot2Labels(Y_test)
idx0 = filterSampleSet(y_test_labels, 0,orig_preds_np)
''' 