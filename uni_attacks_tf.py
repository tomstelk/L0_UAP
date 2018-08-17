

import numpy as np
from six.moves import xrange
import tensorflow as tf


from cleverhans import utils_tf
from cleverhans import utils

_logger = utils.create_logger("cleverhans.attacks.tf")


def computeGradients(x_in, y_in, preds):
    
    nb_classes = int(y_in.shape[-1].value)
    nb_features = int(np.product(x_in.shape[1:]).value)
    
    # create the Jacobian graph
    list_derivatives = []
    for class_ind in xrange(nb_classes):
        derivatives = tf.gradients(preds[:, class_ind], x_in)
        list_derivatives.append(derivatives[0])
    grads = tf.reshape(tf.stack(list_derivatives),
                   shape=[nb_classes, -1, nb_features])
        
        # Compute the Jacobian components
    # To help with the computation later, reshape the target_class
    # and other_class to [nb_classes, -1, 1].
    # The last dimention is added to allow broadcasting later.
    
    target_class = tf.reshape(tf.transpose(y_in, perm=[1, 0]),
                              shape=[nb_classes, -1, 1])
    other_classes = tf.cast(tf.not_equal(target_class, 1), tf.float32)

    
    grads_target = tf.reduce_sum(grads * target_class, axis=0)
    grads_other = tf.reduce_sum(grads * other_classes, axis=0)
    
    return grads_target, grads_other

def saliencyForEachPair(grads_target,grads_other, domain_in,increase,nb_features):
        
    # Remove the already-used input features from the search space
    # Subtract 2 times the maximum value from those value so that
    # they won't be picked later
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = tf.constant(tmp, tf.float32)
    
    increase_coef = (4 * int(increase) - 2) \
            * tf.cast(tf.equal(domain_in, 0), tf.float32)

    target_tmp = grads_target
    target_tmp -= increase_coef \
            * tf.reduce_max(tf.abs(grads_target), axis=1, keep_dims=True)
    target_sum = tf.reshape(target_tmp, shape=[-1, nb_features, 1]) \
            + tf.reshape(target_tmp, shape=[-1, 1, nb_features])

    other_tmp = grads_other
    other_tmp += increase_coef \
            * tf.reduce_max(tf.abs(grads_other), axis=1, keep_dims=True)
    other_sum = tf.reshape(other_tmp, shape=[-1, nb_features, 1]) \
        + tf.reshape(other_tmp, shape=[-1, 1, nb_features])
        
    # Create a mask to only keep features that match conditions
    if increase:
        scores_mask = ((target_sum > 0) & (other_sum < 0))
    else:
        scores_mask = ((target_sum < 0) & (other_sum > 0))
    
    return tf.cast(scores_mask, tf.float32) \
            * (-target_sum * other_sum) * zero_diagonal

def map2L0Ball(x_in, perb_in, y_in, L0_Max, model, clip_min, clip_max):
        
    nb_classes = int(y_in.shape[-1].value)
    nb_features = int(np.product(x_in.shape[1:]).value)
    max_L0_abs = L0_Max*nb_features
    
    def condition(perb2check):
        # Repeat the loop until we have reduced L0 to required level
        currL0=tf.count_nonzero(perb2check, dtype=tf.float32)

        return currL0  > max_L0_abs
    
    def body(perb_in):
        #Add perturbation to original image and compute grads
        x_perb =  tf.clip_by_value(x_in + perb_in, clip_value_min=0., clip_value_max=1.)
        
        preds = model.get_probs(x_perb)
        grads_target, grads_other = computeGradients(x_perb, y_in, preds)
        
        #Nullify perturbations if over L0 limit - domain is perb pixels
        
        search_domain_up = tf.reshape(
                            tf.cast(tf.equal(perb_in, 1.), tf.float32),
                            [-1, nb_features])
    
        search_domain_dn = tf.reshape(
                            tf.cast(tf.equal(perb_in, -1.), tf.float32),
                            [-1, nb_features])
    
        #Calc saliency scores for each pair of pixels, in up and down direction
        scores_up = saliencyForEachPair(grads_target,grads_other, search_domain_up,1,nb_features)
        scores_dn = saliencyForEachPair(grads_target,grads_other, search_domain_dn,0,nb_features)
    
        
        #Get domain of perturbations 
        pair_domain_up = tf.reshape(
            tf.matrix_set_diag(tf.tensordot(tf.transpose(search_domain_up),search_domain_up, axes=1), \
                                        tf.zeros(nb_features)), [-1, nb_features*nb_features])

        pair_domain_dn = tf.reshape(
            tf.matrix_set_diag(tf.tensordot(tf.transpose(search_domain_dn),search_domain_dn,axes=1), \
                                        tf.zeros(nb_features)), [-1, nb_features*nb_features])
        
        #Add 2 everywhere except over the perturbation domains so that min picks out pixels in domain
        score_dom_up = tf.reshape(scores_up,   [-1, nb_features*nb_features]) + 2 - 2*pair_domain_up
        score_dom_dn = tf.reshape(scores_dn,   [-1, nb_features*nb_features]) + 2 - 2*pair_domain_dn
    
        #Get best pixel pair (i.e. the least salient)
        b_upVal=tf.reduce_min(score_dom_up,axis=1)
        b_dnVal=tf.reduce_min(score_dom_dn,axis=1)
    
        b_upArg = tf.argmin(score_dom_up,axis=1)
        b_dnArg = tf.argmin(score_dom_dn,axis=1)
    
        #Select the best between up and down
        best = tf.cond(b_upVal[0]<b_dnVal[0],lambda: b_upArg,lambda: b_dnArg)
        increase = tf.cond(b_upVal[0]<b_dnVal[0],lambda: -1.,lambda: 1.)
            
        p1 = tf.mod(best, nb_features)
        p2 = tf.floordiv(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        p2_one_hot = tf.one_hot(p2, depth=nb_features)
    
    
        to_mod = (p1_one_hot + p2_one_hot)
        to_mod_reshape = tf.reshape(to_mod,
                                    shape=([-1] + x_in.shape[1:].as_list()))* increase
                                
        
        
        perb_out = tf.clip_by_value(to_mod_reshape+perb_in, clip_value_min=clip_min-clip_max, clip_value_max=clip_max)
        
        return perb_out
    
    
    perb_out = tf.while_loop(condition, body, [perb_in], parallel_iterations=1)
    
    return perb_out

def jsma_symbolic_up_dn(x, y_target, model, theta, gamma, clip_min, clip_max, perb_in):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).

    :param x: the input placeholder
    :param y_target: the target tensor
    :param model: a cleverhans.model.Model object.
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :return: a tensor for the adversarial example
    """

    nb_classes = int(y_target.shape[-1].value)
    nb_features = int(np.product(x.shape[1:]).value)

    max_iters = np.floor(nb_features * gamma / 2)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    
    search_domain_up = tf.reshape(
                            tf.cast(x < clip_max, tf.float32),
                            [-1, nb_features])
    
    search_domain_dn = tf.reshape(
                            tf.cast(x > clip_min, tf.float32),
                            [-1, nb_features])
    
    x_perb = tf.clip_by_value(x + perb_in, clip_value_min = clip_min, clip_value_max=clip_max)
    
    # Loop variables
    # x_in: the tensor that holds the latest adversarial outputs that are in
    #       progress.
    # y_in: the tensor for target labels
    # domain_in: the tensor that holds the latest search domain
    # cond_in: the boolean tensor to show if more iteration is needed for
    #          generating adversarial samples
    def condition(x_in, y_in, domain_in_up,domain_in_dn , i_in, cond_in,theta, perb_in):
        # Repeat the loop until we have achieved misclassification or
        # reaches the maximum iterations
        return tf.logical_and(tf.less(i_in, max_iters), cond_in)
    
   

    # Same loop variables as above
    def body(x_in, y_in, domain_in_up,domain_in_dn , i_in, cond_in, theta,perb_in):

        preds = model.get_probs(x_in)
        preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)
        
        
        grads_target, grads_other = computeGradients(x_in, y_in, preds)
        
        # Create a 2D numpy array of scores for each pair of candidate features
        # Considering both up and down perturbations
        scores_up = saliencyForEachPair(grads_target,grads_other, domain_in_up,1,nb_features)
        
        scores_dn = saliencyForEachPair(grads_target,grads_other, domain_in_dn,0,nb_features)
        

            
        # Extract the best pixel pair - for up and down perbs
        b_upVal=tf.reduce_max(
                    tf.reshape(scores_up, shape=[-1, nb_features * nb_features]),
                    axis=1)
        b_dnVal=tf.reduce_max(
                    tf.reshape(scores_dn, shape=[-1, nb_features * nb_features]),
                    axis=1)
        
        b_upArg = tf.argmax(
                    tf.reshape(scores_up, shape=[-1, nb_features * nb_features]),
                    axis=1)
        
        b_dnArg = tf.argmax(
                    tf.reshape(scores_dn, shape=[-1, nb_features * nb_features]),
                    axis=1)
        
        #Select the best between up and down
        best = tf.cond(b_upVal[0]>b_dnVal[0],lambda: b_upArg,lambda: b_dnArg)
        increase = tf.cond(b_upVal[0]>b_dnVal[0],lambda: theta,lambda: -theta)
        domain_in = tf.cond(b_upVal[0]>b_dnVal[0],lambda:domain_in_up,lambda:domain_in_dn)
        
        p1 = tf.mod(best, nb_features)
        p2 = tf.floordiv(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        p2_one_hot = tf.one_hot(p2, depth=nb_features)

        # Check if more modification is needed for each sample
        mod_not_done = tf.equal(tf.reduce_sum(y_in * preds_onehot, axis=1), 0)
        cond = mod_not_done & (tf.reduce_sum(domain_in, axis=1) >= 2)


        cond_float = tf.reshape(tf.cast(cond, tf.float32), shape=[-1, 1])
        to_mod = (p1_one_hot + p2_one_hot) * cond_float
        
        # Update the search domain
        domain_out_up = tf.cond(b_upVal[0]>b_dnVal[0],lambda:domain_in_up - to_mod,lambda:domain_in_up)
        domain_out_dn = tf.cond(b_upVal[0]>b_dnVal[0],lambda:domain_in_dn,lambda:domain_in_dn - to_mod)
        
        # Apply the modification to the images
        to_mod_reshape = tf.reshape(to_mod,
                                    shape=([-1] + x_in.shape[1:].as_list()))* increase
        
        
        x_out = tf.clip_by_value(x_in + to_mod_reshape, clip_value_min=clip_min, clip_value_max=clip_max)
        perb_out = tf.clip_by_value(to_mod_reshape+perb_in, clip_value_min=-1, clip_value_max=1)
        
        # Increase the iterator, and check if all misclassifications are done
        i_out = tf.add(i_in, 1)
        cond_out = tf.reduce_any(cond)

        return x_out, y_in, domain_out_up, domain_out_dn, i_out, cond_out,theta,perb_out
    


    x_adv, _, _, _, _, _,_, perb_out = tf.while_loop(condition, body,
                                      [x_perb, y_target, search_domain_up,search_domain_dn, 0, True,theta, perb_in],
                                      parallel_iterations=1)
    
    return x_adv,perb_out
	
	
def genUniAdvPerb(x_sample, t, model, nb_classes, clip_min, clip_max, L0_Max, theta):
	
    
    #Setup the perb target
    perb_target_one_hot=np.zeros((1, nb_classes), dtype=np.float32)
    perb_target_one_hot[0, t] = 1
    perb_target = tf.constant(perb_target_one_hot)
    
    x_shape = tf.shape(x_sample[0:1])
    sample_Size=np.shape(x_sample)[0]
    perb =tf.zeros(shape=x_shape,dtype=tf.float32)   
    
    for i in range(0,sample_Size):
        
        x = tf.constant(x_sample[i:i+1])
        _, perb = jsma_symbolic_up_dn(x, perb_target, model, theta= theta, 
                                      gamma=0.1, clip_min=0., clip_max=1.,
                                      perb_in=perb)
        
        
        
        perb  = map2L0Ball(x_in=x, perb_in=perb, y_in=perb_target, 
                           L0_Max=L0_Max,model=model,clip_min=0., clip_max=1.)      
        
    return perb
    
    
    
		
	
	