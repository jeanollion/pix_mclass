import tensorflow as tf


class ScheduledDropout(tf.keras.layers.Layer):
    """
    (Spatial) dropout layer with scheduled rate based on training progress.
    """

    def __init__(self,
                 rate: float,  # target rate (min_rate)
                 max_rate: float,  # rate at training start
                 min_progress: float = 0.0,
                 max_progress: float = 1.0,  # When this layer reaches rate
                 spatial: bool = True,  # Use spatial dropout for images
                 power_law: float = 1.0,
                 seed=None,  # Optional: for reproducibility
                 **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.min_rate = rate
        self.max_rate = max_rate
        self.min_progress = min_progress
        self.max_progress = max_progress
        self.spatial = spatial
        self.power_law = power_law
        self.seed = seed

        # Training progress variable [0, 1] - set by callback
        self.progress = None

    def build(self, input_shape):
        super().build(input_shape)

        # Create progress variable (updated by callback)
        self.progress = self.add_weight(
            name='progress',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            dtype=tf.float32
        )
        self.is_3D = self.spatial and len(input_shape) == 5
        # Validate input shape for spatial dropout
        if self.spatial and len(input_shape) not in [4, 5]:
            raise ValueError(
                f"Spatial dropout requires 4D (batch, height, width, channels) or "
                f"5D (batch, depth, height, width, channels) input, got shape {input_shape}"
            )

    def call(self, inputs, training=None):
        if not training:
            return inputs

        current_rate = self.get_current_rate()
        current_rate = tf.cast(current_rate, inputs.dtype)
        if self.spatial:
            # Use tf.nn.dropout with spatial noise_shape for spatial dropout
            input_shape = tf.shape(inputs)

            if not self.is_3D:
                # 2D convolutions: drop (batch, 1, 1, channels)
                noise_shape = [input_shape[0], 1, 1, input_shape[3]]
            else:
                # 3D convolutions: drop (batch, 1, 1, 1, channels)
                noise_shape = [input_shape[0], 1, 1, 1, input_shape[4]]

            return tf.nn.dropout(
                inputs,
                rate=current_rate,
                noise_shape=noise_shape,
                seed=self.seed
            ) * (1 - current_rate) # unscaled dropout

        else:
            # Regular dropout (no noise_shape = element-wise dropout)
            return tf.nn.dropout(
                inputs,
                rate=current_rate,
                seed=self.seed
            ) * (1 - current_rate)


    def set_progress(self, progress_value):
        """Set global training progress [0, 1] - called by callback"""
        self.progress.assign(tf.clip_by_value(progress_value, 0.0, 1.0))

    def get_current_rate(self):
        """Get current dropout rate"""
        if self.progress is None:
            return self.max_rate

        # Calculate layer-specific progress
        # Cast to float32 explicitly to handle mixed precision
        progress_val = tf.cast(self.progress, tf.float32) - tf.cast(self.min_progress, tf.float32)
        progress_norm = tf.cast(self.max_progress - self.min_progress, tf.float32)
        layer_progress = tf.maximum(0.0, tf.minimum(1.0, progress_val / progress_norm))

        # Interpolate from max_rate to min_rate
        max_rate_val = tf.cast(self.max_rate, tf.float32)
        min_rate_val = tf.cast(self.min_rate, tf.float32)
        current_rate = max_rate_val - (max_rate_val - min_rate_val) * tf.math.pow(layer_progress, self.power_law)

        return current_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "rate": float(self.min_rate),
            "max_rate": float(self.max_rate),
            "min_progress": float(self.min_progress),
            "max_progress": float(self.max_progress),
            "spatial": self.spatial,
            "power_law": self.power_law,
            "seed": self.seed
        })
        return config


class ScheduledGradientWeight(tf.keras.layers.Layer):
    """
    Layer that applies scheduled gradient weighting to skip connections.
    Forward pass: unchanged (information flows normally)
    Backward pass: gradients are scaled by a scheduled weight
    """

    def __init__(self,
                 min_weight: float = 0.0,  # Initial gradient weight (training start)
                 max_weight: float = 1.0,  # Final gradient weight (target)
                 min_progress: float = 0.0,
                 max_progress: float = 1.0,  # When this layer reaches max_weight
                 power_law: float = 1.0,
                 **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_progress = min_progress
        self.max_progress = max_progress
        self.power_law=power_law

        # Training progress variable [0, 1] - set by callback
        self.progress = None

    def build(self, input_shape):
        super().build(input_shape)

        # Create progress variable (updated by callback)
        # Initialize to 0.0 (training start)
        self.progress = self.add_weight(
            name='progress',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            dtype=tf.float32
        )

    @tf.custom_gradient
    def _weight_gradient(self, x, weight):
        """
        Forward pass: return input unchanged
        Backward pass: scale gradients by weight
        """

        def grad(dy):
            # Handle different gradient types
            if isinstance(dy, tuple):
                return tuple(y * tf.cast(weight, y.dtype) for y in dy), None
            elif isinstance(dy, list):
                return [y * tf.cast(weight, y.dtype) for y in dy], None
            else:
                return dy * tf.cast(weight, dy.dtype), None

        return x, grad

    def call(self, inputs, training=None):
        if not training:
            return inputs
        current_weight = self.get_current_weight()
        return self._weight_gradient(inputs, current_weight)

    def set_progress(self, progress_value):
        """Set global training progress [0, 1] - called by callback"""
        self.progress.assign(tf.clip_by_value(progress_value, 0.0, 1.0))

    def get_current_weight(self):
        """Get current gradient weight"""
        if self.progress is None:
            return self.min_weight

        # Calculate layer-specific progress
        # Cast to float32 explicitly to handle mixed precision
        progress_val = tf.cast(self.progress, tf.float32) -  tf.cast(self.min_progress, tf.float32)
        progress_norm = tf.cast(self.max_progress - self.min_progress, tf.float32)
        layer_progress = tf.maximum(0.0, tf.minimum(1.0, progress_val / progress_norm))

        # Interpolate from min_weight to max_weight (inverse of dropout)
        min_weight_val = tf.cast(self.min_weight, tf.float32)
        max_weight_val = tf.cast(self.max_weight, tf.float32)
        current_weight = min_weight_val + (max_weight_val - min_weight_val) * tf.math.pow(layer_progress, self.power_law)

        return current_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_weight": float(self.min_weight),
            "max_weight": float(self.max_weight),
            "min_progress": float(self.min_progress),
            "max_progress": float(self.max_progress),
            "power_law": float(self.power_law)
        })
        return config


class ResidualGradientLimiter(tf.keras.layers.Layer):
    """
    Limits skip path gradients relative to main path gradients.
    Only reduces skip gradients (never amplifies them).

    Usage:
        limiter = ResidualGradientLimiter()
        limited_skip, main = limiter([skip_path, main_path], training=True)
    """

    def __init__(self,
                 max_ratio: float = 1.0,
                 epsilon: float = 1e-5,  # Numerical stability
                 **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.max_ratio=max_ratio
        self.epsilon = epsilon

    def build(self, input_shape):
        """
        input_shape is a list: [skip_shape, main_shape]
        """
        super().build(input_shape)

        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"ResidualGradientLimiter expects exactly 2 inputs [skip, main], "
                f"got {len(input_shape) if isinstance(input_shape, list) else 1}"
            )

    @tf.custom_gradient
    def _limit_gradients(self, x):
        skip, main = x
        """
        Forward: return inputs unchanged
        Backward: limit skip gradients to match main gradient magnitude (only reduce)

        Args:
            skip: skip connection tensor
            main: main path tensor
        """
        def grad(dy_skip, dy_main):
            main_grad_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.cast(dy_main, tf.float32) ** 2), self.epsilon))
            if dy_skip is None:
                return dy_skip, dy_main
            skip_grad_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.cast(dy_skip, tf.float32) ** 2), self.epsilon))

            # Compute scaling factor to limit res path gradient to the scale of main path i.e. |dy_skip_scaled|| <= max_ratio * ||dy_main||
            scale_factor = tf.minimum(  self.max_ratio * main_grad_norm / tf.maximum(skip_grad_norm, self.epsilon), 1.0 ) # only limit
            return dy_skip * tf.cast(scale_factor, dy_skip.dtype), dy_main
        return [skip, main], grad

    def call(self, inputs, training=None):
        """
        Args:
            inputs: list of [skip_tensor, main_tensor]
            training: whether in training mode

        Returns:
            list of [limited_skip, main]
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"ResidualGradientLimiter expects a list/tuple of 2 tensors [skip, main], "
                f"got {type(inputs)}"
            )

        if not training: # During inference, no gradient limiting
            return inputs
        return self._limit_gradients(inputs)

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

