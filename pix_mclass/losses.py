import warnings
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import dataset_iterator.helpers as dih

def get_class_counts(dataset, channel_keyword:str= "classes"):
    histo, vmin, bins = dih.get_histogram(dataset, channel_keyword, bins=None, bin_size=1, return_min_and_bin_size=True, max_decimation_factor=3)
    histo = histo.astype(np.float64)
    bck_count = 0
    if vmin == 0: # remove non-annotated pixels
        bck_count = histo[0]
        histo = histo[1:]
    sum = np.sum(histo)
    if sum == 0:
        warnings.warn(f"no pixels are annotated in dataset: {dataset}")
        return np.zeros_like(histo)
    if np.any(histo == 0):
        idx = np.nonzero(histo==0)
        warnings.warn(f"classes {list(idx[0])} have no annotated pixels on dataset: {dataset}")
    return histo, bck_count

def get_weighted_sparse_categorical_crossentropy(weights, dtype='float32', **cce_kwargs):
    weights_cast = np.array(weights).astype(dtype)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(**cce_kwargs)
    wfun = get_category_sample_weights_fun(weights_cast, dtype=dtype)
    def loss_func(true, pred):
        true, wm = tf.split(true, 2, axis=-1)
        weights = wfun(true) * wm[...,0]
        return cce(true, pred, sample_weight=weights)
    return loss_func

def get_weighted_sparse_categorical_tempered_focal_loss(weights, dtype='float32', **loss_kwargs):
    weights_cast = np.array(weights).astype(dtype)
    loss = TemperedFocalCrossEntropy(**loss_kwargs)
    wfun = get_category_sample_weights_fun(weights_cast, dtype=dtype)
    def loss_func(true, pred):
        true, wm = tf.split(true, 2, axis=-1)
        weights = wfun(true) * wm[...,0]
        return loss(true, pred, sample_weight=weights)
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


class TemperedFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, temperature: float = 1.0, focal_weight = 2.0,
                 label_smoothing: float = 0, sparse:bool=True, **kwargs):
        """
        Tempered Focal Cross-Entropy with Label Smoothing for multi-class classification.
        Combines gradient stability (tempering), hard example mining (focal),
        and regularization (label smoothing).

        Args:
            temperature: Tempering parameter (t ≥ 1). Controls gradient bounding.
                        - t=1.0 → standard focal loss (unbounded gradients, as in classical categorical cross entropy)
                        - t=2.0 → moderate bounding
                        - t=3.0+ → strong bounding (very stable, may slow learning)

            focal_weight: Focusing parameter (γ ≥ 0). Controls hard example emphasis. can be a list / tuple -> one value for each class
                   γ=0.0 → classical cross entropy (no focal effect)
                   γ=1.0 → mild focus on hard examples
                   γ=2.0 → standard focal (recommended start)
                   γ=5.0 → extreme focus (for very imbalanced data)

            label_smoothing: Smoothing parameter (0 ≤ ε < 1). Regularization strength.
                            ε=0.0 → no smoothing (hard labels)
                            ε=0.1 → typical for ImageNet (recommended start)
                            ε=0.2 → stronger regularization
                            Effect: y_smooth = y * (1-ε) + ε/K where K=num_classes

                            Benefits:
                            - Prevents overconfidence (probabilities ≠ 0 or 1)
                            - Improves calibration (predicted probs match true frequencies)
                            - Acts as regularization (reduces overfitting)
                            - Better generalization on test data

                            When useful:
                            - Models prone to overconfidence
                            - Limited training data
                            - Noisy labels
                            - When calibration matters (e.g., medical, finance)

                            Trade-offs:
                            - May slightly hurt training accuracy
                            - Improves test accuracy & calibration
                            - Can conflict with focal loss (both modify targets)
        """
        self.temperature = float(temperature)
        if focal_weight is None or (isinstance(focal_weight, (float, int)) and focal_weight == 0):
            self.focal_weight = None
        else:
            self.focal_weight = np.atleast_1d(np.array(focal_weight, dtype=np.float32))
        self.label_smoothing = float(label_smoothing)
        print(f"CE: label_smoothing={self.label_smoothing} focal weight: {self.focal_weight} temperature: {self.temperature}")
        assert temperature >= 1, f"temperature must be >=1, got {temperature}"
        assert focal_weight is None or np.all(focal_weight >= 0), f"gamma must be >=0, got {focal_weight}"
        assert 0 <= label_smoothing < 1, f"label_smoothing must be in [0,1), got {label_smoothing}"
        self.sparse = sparse
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: One-hot encoded labels, shape (batch_size, num_classes)
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        if self.sparse: # Convert sparse labels to one-hot encoded labels
            num_classes = tf.shape(y_pred)[-1]
            y_true = tf.one_hot(tf.cast(tf.squeeze(y_true, -1), tf.int32), depth=num_classes, dtype=y_pred.dtype)

        # Apply label smoothing: y_smooth = y * (1-ε) + ε/K
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = y_true * (1. - self.label_smoothing) + self.label_smoothing / num_classes

        # Tempered log: (p^(1-t) - 1) / (1-t)
        if self.temperature > 1:
            tempered_log = (tf.pow(y_pred, 1. - self.temperature) - 1.) / (1. - self.temperature)
        else:
            tempered_log = tf.math.log(y_pred)

        # Focal weight: (1 - p)^gamma
        # Note: With label smoothing, focal effect is slightly reduced
        # since targets are no longer pure 0/1
        if self.focal_weight is not None:
            if len(self.focal_weight) == 1:
                focal_weight = tf.pow(1. - y_pred, tf.constant(self.focal_weight[0], dtype=y_pred.dtype))
            else: # per class gamma
                weight_tensor = tf.constant(self.focal_weight, dtype=y_pred.dtype)
                weight_tensor = tf.reshape(weight_tensor, [1] * (len(y_pred.shape) - 1) + [-1])
                focal_weight = tf.pow(1. - y_pred, weight_tensor)
        else:
            focal_weight = tf.cast(1, y_true.dtype)

        # Combined loss
        loss = - focal_weight * y_true * tempered_log
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'focal_weight': list(self.focal_weight) if self.focal_weight is not None else None,
            'label_smoothing': self.label_smoothing,
            "sparse": self.sparse
        })
        return config
