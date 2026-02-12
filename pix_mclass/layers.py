import tensorflow as tf

class ResidualGradientLimiter(tf.keras.layers.Layer):
    """
    Limits skip path gradients relative to main path gradients.
    Only reduces skip gradients (never amplifies them).

    Usage:
        limiter = ResidualGradientLimiter()
        limited_skip, main = limiter(x, training=True)
    """

    def __init__(self,
                 max_ratio: float = 1.0,
                 epsilon: float = 1e-5,
                 **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.max_ratio = max_ratio
        self.epsilon = epsilon


    @tf.custom_gradient
    def _limit_gradients(self, x):
        skip, main = x

        def grad(dy_skip, dy_main):
            main_grad_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.cast(dy_main, tf.float32) ** 2), self.epsilon))
            if dy_skip is None:
                return dy_skip, dy_main
            skip_grad_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.cast(dy_skip, tf.float32) ** 2), self.epsilon))
            scale_factor = tf.minimum(
                self.max_ratio * main_grad_norm / tf.maximum(skip_grad_norm, self.epsilon),
                1.0
            )
            return dy_skip * tf.cast(scale_factor, dy_skip.dtype), dy_main

        return [skip, main], grad


    def call(self, inputs, training=None):
        """
        Args:
            inputs: single tensor to be split into skip and main paths
            training: whether in training mode

        Returns:
            list of [limited_skip, main]
        """
        # Split the input into two paths using tf.identity
        res = tf.identity(inputs, name=f"{self.name}_residual")
        x = inputs

        if not training:
            return [res, x]

        return self._limit_gradients([res, x])

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_ratio": float(self.max_ratio),
            "epsilon": float(self.epsilon),
        })
        return config


@tf.keras.utils.register_keras_serializable(package='Custom', name='HybridThresholdL2Regularizer')
class HybridThresholdL2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self,
                 directional_threshold:float=2.,
                 directional_strength:float=1e-3,
                 elementwise_threshold:float=10.0,
                 elementwise_strength:float=1e-4,
                 axis=None):
        self.directional_threshold = directional_threshold
        self.directional_strength = directional_strength
        self.elementwise_threshold = elementwise_threshold
        self.elementwise_strength = elementwise_strength
        self.axis=axis

    def __call__(self, weights):
        norm_axis = tuple(range(weights.shape.rank - 1)) if self.axis is None else (self.axis if isinstance(self.axis, (tuple, list)) else (self.axis,) )
        if self.directional_strength > 0 and len(norm_axis) > 0:
            norms = tf.sqrt(tf.reduce_sum(tf.square(weights), axis=norm_axis))
            directional_excess = tf.nn.relu(norms - self.directional_threshold)
            directional_penalty = tf.reduce_sum(tf.square(directional_excess))
        else:
            directional_penalty = 0
        if self.elementwise_strength > 0:
            abs_weights = tf.abs(weights)
            elementwise_excess = tf.nn.relu(abs_weights - self.elementwise_threshold)
            elementwise_penalty = tf.reduce_sum(tf.square(elementwise_excess))
        else:
            elementwise_penalty = 0
        return self.directional_strength * directional_penalty + self.elementwise_strength * elementwise_penalty

    def get_config(self):
        return {
            "directional_threshold":self.directional_threshold,
            "directional_strength":self.directional_strength,
            "elementwise_threshold":self.elementwise_threshold,
            "elementwise_strength":self.elementwise_strength,
            "axis":self.axis
        }

