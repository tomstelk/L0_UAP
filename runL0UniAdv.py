

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.attacks_tf import jsma_symbolic
import matplotlib.pyplot as plt
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_mnist import data_mnist
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from uni_attacks_tf import genUniAdvPerb
import scipy.misc
import L0_Utils



imagesforDocDir = "C:/Users/user/Documents/NN Work/UAN/ImagesForDoc/"


train_start=0
train_end=60000
test_start=0
test_end=10000
perb_target = 2
clip_min = 0.
clip_max = 1.
nb_classes=10
res_dir = './uniAdvRes/'
cnnModelFile = "../UAN/savedmodels/basic_cnn_MNIST/basic_cnn.ckpt"
advTarget=0
advSampleSetSize=10

# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                              train_end=train_start+train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

xDim = np.shape(X_train)[1:3]
tf.reset_default_graph()

orig_preds = L0_Utils.getModelPreds(X_train, make_basic_cnn, cnnModelFile)
filterSample=L0_Utils.filterSampleSet(Y_train,advTarget,orig_preds)
                                      
advGenSample=X_train[filterSample][0:advSampleSetSize]




tf.reset_default_graph()
cnn_model=make_basic_cnn()
saver=tf.train.Saver()
perb=genUniAdvPerb(x_sample =advGenSample, t=advTarget, model=cnn_model, 
                   nb_classes=10, clip_min=0., clip_max=1., L0_Max=0.01, 
                   theta=1.0)

with tf.Session() as sess:
    saver.restore(sess,cnnModelFile)
    perb_np = perb.eval()
    

'''


tf.reset_default_graph()

perb_x=np.clip(advGenSample +  perb_np,0,1)
orig_preds_sample=np.argmax(Y_train[filterSample], axis=1)
perb_preds_sample = L0_Utils.getModelPreds(perb_x, make_basic_cnn, cnnModelFile)

fig, (ax1) = plt.subplots(1, 1)

ax1.imshow(perb_np[0,:,:,0])
ax1.axis('off')

for i in range(len(advGenSample)):
    x=advGenSample[i,:,:,0]
    p=perb_x[i,:,:,0]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(x)
    ax1.axis('off')

    ax2.imshow(p)
    ax2.axis('off')
    fig.suptitle("orig pred: " + str(orig_preds_sample[i]) + ", " + "perb pred: " +str(perb_preds_sample[i]))
    
    if orig_preds_sample[i] !=perb_preds_sample[i]:
        im_orig = imagesforDocDir + "L0_orig_" + str(orig_preds_sample[i]) + ".png"
        im_perb = imagesforDocDir + "L0_orig_" + str(orig_preds_sample[i]) + "_perb_" + str(perb_preds_sample[i]) + ".png"
        plt.imsave(im_orig,x)
        plt.imsave(im_perb,p)
'''