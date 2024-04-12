import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import dataset_iterator.helpers as dih

def get_class_weights(dataset, channel_keyword:str="classes"):
    histo, vmin, bins = dih.get_histogram(dataset, channel_keyword, bins=None, bin_size=1, return_min_and_bin_size=True)
    histo = histo.astype(np.float64)
    if vmin == 0: # remove non-annotated pixels
        histo = histo[1:]
    sum = np.sum(histo)
    #print(f"histo: {histo} sum: {sum}, vmin: {vmin}")
    return sum / histo

def weighted_sparse_categorical_crossentropy(weights, dtype='float32', **cce_kwargs):
    weights_cast = np.array(weights).astype(dtype)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(**cce_kwargs)
    wfun = get_category_sample_weights_fun(weights_cast, dtype=dtype)
    def loss_func(true, pred):
        true, wm = tf.split(true, 2, axis=-1)
        weights = wfun(true) * wm[...,0]
        return cce(true, pred, sample_weight=weights)
    return loss_func

def get_category_sample_weights_fun(weights_list, axis=-1, sparse=True, dtype='float32'):
    weights_list_cast = np.array(weights_list).astype(dtype)
    class_indices = np.array([i for i in range(len(weights_list_cast))]).astype(dtype)
    def weight_fun(true):
        if sparse:
            class_selectors = K.squeeze(true, axis=axis)
        else:
            class_selectors = K.argmax(true, axis=axis)

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        class_selectors = [K.equal(i, class_selectors) for i in class_indices]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        class_selectors = [tf.cast(x, dtype) for x in class_selectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(class_selectors, weights_list_cast)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]

        #make sure your original_loss_func only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        return weight_multiplier
    return weight_fun
