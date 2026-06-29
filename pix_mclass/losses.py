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
    def __init__(self, sparse:bool=True, focal_weight = 0.0, temperature: float = 0.0, pseudo_huber: float = 0.0, label_smoothing: float = 0.01, **kwargs):
        """
        Tempered Focal Cross-Entropy with Label Smoothing for multi-class classification.
        Combines hard example mining (focal), gradient bounding (temperature), and
        regularization (label smoothing).

        Args:
            focal_weight: Focusing parameter (γ ≥ 0). Controls hard example emphasis. can be a list / tuple -> one value for each class
                   γ=0.0 → tempered CE (no focal effect)
                   γ=1.0 → mild focus on hard examples
                   γ=2.0 → standard focal (recommended start)
                   γ=5.0 → extreme focus (for very imbalanced data)

            temperature: Tempering parameter (0 ≤ t < 1). Replaces log(p) with the tempered
                   logarithm log_t(p) = (p^t - 1) / t, whose gradient p^(t-1) decays
                   instead of blowing up like 1/p. This bounds the per-pixel loss on
                   confident-wrong / hard pixels (loss ≤ 1/t for p→0), so a few ambiguous
                   or mislabeled examples can no longer emit the huge gradients that
                   destabilize mixed-precision training. Higher t = tighter bound (1/t → 1
                   as t → 1) and more robustness to label noise, but may slow learning of
                   confident decisions.
                   t=0.0 → standard cross entropy (log, unbounded gradient)
                   t=0.1 → mild bounding (loss capped at -10 for p→0)
                   t=0.5 → strong bounding (loss capped at -2, very stable, may slow learning)

            pseudo_huber: Pseudo-Huber strength c (0 ≤ c ≤ 1), in units of chance level 1/K.
                   Internally floors the log by a = c/K (K = #classes): log(p) → log(p + c/K),
                   the smooth pseudo-Huber analog of cross entropy — CE for p ≫ c/K, linear
                   (MAE / L1, constant gradient) for p ≪ c/K, so a confident-wrong term's
                   gradient is bounded by 1/a = K/c instead of CE's unbounded 1/p. c=0 → CE.

                   Why c and not the raw floor a: 1/K (chance) is the natural probability unit
                   — the analog of the pixel unit for a regression pseudo-Huber delta — so c
                   is K-independent and reads directly as "the fraction of chance below which
                   an example is treated as an outlier". At chance p=1/K the CE gradient is K
                   and the floor caps it at K/c, so c<1 caps only worse-than-chance preds.
                     c ≈ 0.1–0.3 → cap only well-below-chance preds (clear mislabels), CE
                                   elsewhere — recommended
                     c = 1        → transition exactly at chance (more robust)
                   Equivalent readings: max gradient = K/c (lower tail; + γ|log(c/K)| with
                   focal γ), max loss ≈ log(K/c).

                   Applied to the whole probability vector, so WITH label_smoothing>0 the
                   smoothed wrong-class terms (which diverge as p→0 under over-confidence) are
                   floored too — capping the OVER-confident end as well; without smoothing only
                   the confident-wrong (true-class) tail is capped. Keep c ≲ ε (the smoothing
                   target is ε/K vs the floor c/K) to preserve smoothing's anti-overconfidence
                   push if overflow is the concern. Alternative robustifier to `temperature`
                   (temperature bounds the loss *value*, pseudo_huber the *gradient*); use one.

            label_smoothing: Smoothing parameter (0 ≤ ε < 1). y_smooth = y*(1-ε) + ε/K
                            (K=num_classes). Prevents over-confidence, improves calibration,
                            and bounds logit growth (so it also fights the fp16 logit-overflow).

                            Choosing ε — natural anchor: ε ≈ label noise / ambiguity rate ρ
                            (don't demand more confidence than the labels deserve; analog of
                            setting a regression Huber delta at the noise level). Three reads
                            of the same ε:
                              - confidence ceiling: optimum p_true ≈ 1-ε  (K-independent)
                              - logit bound: converged logit gap ≈ ln(K/ε); to keep gap ≤ Z
                                use ε ≥ K·e^{-Z} (logarithmic -> even tiny ε bounds it)
                              - per-wrong-class floor: ε/K
                            By purpose:
                              - overflow / logit control: tiny ε suffices (ε≈0.01 -> gap≈5.7);
                                use the smallest that bounds, to perturb labels least
                              - calibration / noise robustness: ε ≈ ρ (typ. 0.01-0.1)
                              - small K (e.g. 3): ImageNet's 0.1 (tuned for K=1000) is strong;
                                prefer 0.01-0.05
                            Match to the logit soft-cap so they cooperate: the soft-cap bounds
                            each logit to ~±c_softcap, so the max gap is ~2·c_softcap; the
                            smoothed optimum gap is ln(K/ε), so it stays inside the cap when
                            ε ≳ K·e^{-2·c_softcap} (c_softcap=4,K=3 -> ε ≳ 1e-3; ε≈0.01 sits
                            comfortably inside). Much smaller ε -> smoothing wants a gap beyond
                            2·c_softcap and tanh fights it; much larger -> cap rarely engages.
                            Practical range ~[0.005, 0.2]; don't stack heavy smoothing with
                            heavy focal/temperature/pseudo_huber (all damp confidence ->
                            under-training); too large -> genuinely under-confident (≤ 1-ε).
        """
        self.sparse=sparse
        if focal_weight is None or (isinstance(focal_weight, (float, int)) and focal_weight == 0):
            self.focal_weight = None
        else:
            self.focal_weight = np.atleast_1d(np.array(focal_weight, dtype=np.float32))
        self.temperature = float(temperature)
        self.pseudo_huber = float(pseudo_huber)
        self.label_smoothing = float(label_smoothing)
        print(f"Cat. Loss: focal weight: {self.focal_weight} label smoothing: {self.label_smoothing} temperature: {self.temperature} pseudo_huber: {self.pseudo_huber}")
        assert self.focal_weight is None or np.all(self.focal_weight >= 0), f"gamma must be >=0, got {focal_weight}"
        assert 1 > self.temperature >= 0, f"temperature must be >=0 and <1, got {temperature}"
        assert 0 <= self.pseudo_huber <= 1, f"pseudo_huber (c, in units of chance 1/K) must be in [0,1], got {pseudo_huber}"
        assert 0 <= label_smoothing < 1, f"label_smoothing must be in [0,1), got {label_smoothing}"
        # Tempered log: log_t(p) = (p^t - 1)/t, gradient p^(t-1). t=0 is the 0/0 limit
        # = standard log (handled separately below). pseudo_huber floors the log by a=c/K
        # (c=self.pseudo_huber, K=#classes): log(p+a) -> gradient bounded by 1/a=K/c.

        super().__init__(**kwargs)

    def _tempered_log(self, p, floor=0.):
        # Pseudo-Huber floor a (=c/K): log(p+a) stays CE for p>>a, linear (bounded grad 1/a) for p<<a.
        if self.pseudo_huber > 0.:
            p = p + tf.cast(floor, p.dtype)
        # t=0: standard natural log (unbounded gradient). 0<t<1: bounded tempered log (loss <= 1/t).
        if self.temperature == 0.:
            return tf.math.log(p)
        e = tf.cast(self.temperature, p.dtype)
        return (tf.pow(p, e) - 1.) / e

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: One-hot encoded labels, shape (batch_size, (Y, X), num_classes) if sparse else dense labels (batch_size, (Y, X), 1)
            y_pred: Predicted probabilities, shape (batch_size, (Y, X), num_classes)
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
        if self.sparse: # Convert sparse labels to one-hot encoded labels
            y_true = tf.one_hot(tf.cast(tf.squeeze(y_true, -1), tf.int32), depth=tf.cast(num_classes, tf.int32), dtype=y_pred.dtype)

        # Apply label smoothing: y_smooth = y * (1-ε) + ε/K
        if self.label_smoothing > 0:
            y_true = y_true * (1. - self.label_smoothing) + self.label_smoothing / num_classes

        # Focal weight: (1 - p)^gamma
        # Note: With label smoothing, focal effect is slightly reduced since targets are no longer pure 0/1
        if self.focal_weight is not None:
            if len(self.focal_weight) == 1:
                focal_weight = tf.pow(1. - y_pred, tf.constant(self.focal_weight[0], dtype=y_pred.dtype))
            else: # per class gamma
                weight_tensor = tf.constant(self.focal_weight, dtype=y_pred.dtype)
                weight_tensor = tf.reshape(weight_tensor, [1] * (len(y_pred.shape) - 1) + [-1])
                focal_weight = tf.pow(1. - y_pred, weight_tensor)
        else:
            focal_weight = tf.cast(1, y_true.dtype)

        # Pseudo-Huber floor a = c/K (K = #classes), in units of chance level; 0 if disabled
        floor = self.pseudo_huber / tf.cast(tf.shape(y_pred)[-1], y_pred.dtype) if self.pseudo_huber > 0 else 0.
        # Combined loss (tempered log / pseudo-Huber floor bound the per-pixel loss / gradient)
        loss = - focal_weight * y_true * self._tempered_log(y_pred, floor)
        # collapse the class axis so the result has the same rank as the per-pixel sample_weight
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'pseudo_huber': self.pseudo_huber,
            'focal_weight': list(self.focal_weight) if self.focal_weight is not None else None,
            'label_smoothing': self.label_smoothing,
            'sparse': self.sparse
        })
        return config
