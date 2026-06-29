import tensorflow as tf


def get_group_norm_groups(num_channels:int, target:int=32, min_per_group:int=4, warn_on_degenerate:bool=True):
    """Heuristic to pick number of channel groups for group-style normalization.

    Aim for `target` groups (32 per GN paper), constrained by:
      - groups must divide num_channels exactly,
      - each group must contain at least `min_per_group` channels.

    Examples (target=32, min_per_group=4):
      C=16  -> 4    (16/4 = 4 ch/group)
      C=32  -> 8    (32/8 = 4)
      C=64  -> 16   (64/16 = 4)
      C=96  -> 24   (96/24 = 4)
      C=128 -> 32   (128/32 = 4)
      C=192 -> 32   (192/32 = 6)
      C=256 -> 32   (256/32 = 8)
      C=512 -> 32   (512/32 = 16)
    For C <= min_per_group, returns 1.
    For prime / non-decomposable C > min_per_group, also returns 1; in that case
    a UserWarning is emitted (silenceable via warn_on_degenerate=False).
    """
    if num_channels <= min_per_group:
        return 1
    max_groups = num_channels // min_per_group
    ideal = min(target, max_groups)
    # Largest divisor of num_channels in [2, ideal] (G=1 is the fallback below).
    for g in range(ideal, 1, -1):
        if num_channels % g == 0:
            return g
    # Fallback: G=1 (LN). Only warn when ideal>=2 — i.e. there *should* have been
    # room for a non-trivial divisor, but num_channels is prime / awkward.
    if warn_on_degenerate and ideal >= 2:
        import warnings
        warnings.warn(
            f"WindowGroupNormalization group heuristic degenerated to G=1 (LayerNorm) for "
            f"num_channels={num_channels}: no divisor in [2, {ideal}] satisfies "
            f"min_per_group={min_per_group}. Consider using a composite channel "
            f"count (e.g. a multiple of 8 or 16) if you actually want GN behavior.",
            stacklevel=2,
        )
    return 1


class WindowGroupNormalization(tf.keras.layers.Layer):
    """Per-(sample, group) normalization with locally-pooled stats over a fixed
    spatial window. Size-invariant at inference: a 1024x1024 image is normalized
    identically to a 256x256 tile because the window slides across the spatial
    dims and each pixel sees only its own local context.

    Supports both 2D (input rank 4: B, Y, X, C) and 3D (input rank 5: B, Z, Y, X, C)
    seamlessly: the spatial pooling op is selected at build time from the
    static input rank.

    Args:
        groups: number of channel groups (must divide C). If None or 0, the
            heuristic `get_group_norm_groups` is applied to pick a sensible G.
        window_size: int OR tuple/list. Spatial window for local stat pooling.
            - int: expanded to all spatial dims (e.g. 32 -> (32, 32) in 2D,
              (32, 32, 32) in 3D).
            - tuple/list: must match number of spatial dims. For 3D anisotropic
              data, pass e.g. (1, 32, 32) for per-Z-slice normalization.
        epsilon: numerical stabilizer.
        center, scale: include affine beta/gamma.
    """
    def __init__(self, groups=None, window_size=32, epsilon=1e-3,
                 center=True, scale=True,
                 name="WindowGroupNormalization", **kwargs):
        super().__init__(name=name, **kwargs)
        self.groups = groups
        self.window_size = window_size  # validated in build
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "groups": self.groups,
            "window_size": self.window_size,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale
        })
        return config

    def build(self, input_shape):
        try:
            input_shape = input_shape.as_list()
        except AttributeError:
            pass
        ndim = len(input_shape)
        if ndim not in (4, 5):
            raise ValueError(
                f"WindowGroupNormalization expects rank 4 (B,Y,X,C) or 5 (B,Z,Y,X,C) input, "
                f"got rank {ndim}"
            )
        self._tridim = (ndim == 5)
        C = int(input_shape[-1])
        # Pick groups via heuristic if not provided
        if self.groups is None or self.groups == 0:
            self.groups = get_group_norm_groups(C)
        if C % self.groups != 0:
            raise ValueError(f"groups={self.groups} must divide channels={C}")
        self._channels_per_group = C // self.groups

        # Expand window_size to match spatial dims
        n_spatial = ndim - 2  # 2 for 2D, 3 for 3D
        if isinstance(self.window_size, int):
            ws = [self.window_size] * n_spatial
        else:
            ws = list(self.window_size)
            if len(ws) != n_spatial:
                raise ValueError(
                    f"window_size has {len(ws)} elements but input rank {ndim} "
                    f"expects {n_spatial} spatial dimensions"
                )
        # Clamp window per dim to at most the size of that spatial axis if known
        for i in range(n_spatial):
            dim = input_shape[1 + i]
            if dim is not None and ws[i] > dim:
                ws[i] = dim
        self._window = ws

        if self.scale:
            self.gamma = self.add_weight(name="gamma", shape=(C,), initializer="ones", dtype="float32", autocast=False)
        if self.center:
            self.beta = self.add_weight(name="beta", shape=(C,), initializer="zeros", dtype="float32", autocast=False)
        # Broadcast shape for gamma/beta against (B, [Z,] Y, X, G, Cg) — built once.
        self._vars_shape = [1] * (n_spatial + 1) + [self.groups, self._channels_per_group]
        super().build(input_shape)

    def call(self, inputs):
        # Stats (mean, variance) are computed in fp32 for numerical stability
        # under mixed_float16 — matches what BN / LN do internally. Cast back to
        # the input dtype (typically fp16) just before returning.
        x = tf.cast(inputs, tf.float32)
        static_shape = inputs.shape.as_list()
        n_spatial = len(self._window)

        # For each spatial axis, decide whether the window covers the whole extent.
        # When dim is unknown (None) we conservatively treat it as local.
        is_global = []
        for i in range(n_spatial):
            dim = static_shape[1 + i]
            is_global.append(dim is not None and self._window[i] >= dim)

        x_shape = tf.shape(x)
        if self._tridim:
            new_shape = tf.concat([x_shape[:4], [self.groups, self._channels_per_group]], axis=0)
        else:
            new_shape = tf.concat([x_shape[:3], [self.groups, self._channels_per_group]], axis=0)

        if all(is_global):
            # Fast global path: behaves exactly like standard GroupNormalization.
            # No padding, no avg_pool: just reduce over all spatial axes + Cg.
            x_g = tf.reshape(x, new_shape)
            if self._tridim:
                reduce_axes = [1, 2, 3, -1]  # Z, Y, X, Cg
            else:
                reduce_axes = [1, 2, -1]     # Y, X, Cg
            m  = tf.reduce_mean(x_g,         axis=reduce_axes, keepdims=True)
            ms = tf.reduce_mean(x_g * x_g,   axis=reduce_axes, keepdims=True)
            var = tf.maximum(ms - m * m, tf.cast(0.0, m.dtype))
            inv = tf.math.rsqrt(var + tf.cast(self.epsilon, var.dtype))
            if self.scale:
                inv = inv * tf.reshape(self.gamma, self._vars_shape)
            res = -m * inv
            if self.center:
                res = res + tf.reshape(self.beta, self._vars_shape)
            x_g = x_g * inv + res
            out = tf.reshape(x_g, x_shape)
        else:
            # Local / mixed path: locally-pooled stats via SAME avg_pool. SAME slides the
            # fixed window and at borders averages only the in-bounds elements (divides by
            # the valid count), i.e. the window is clamped at the edges. No manual padding:
            # works for any input size (including smaller than the window) and preserves
            # spatial shape. For axes where window >= dim we pool with size 1 (no-op) and
            # reduce_mean over them afterwards so they behave globally.
            effective_window = [
                1 if is_global[i] else self._window[i] for i in range(n_spatial)
            ]
            x2 = x * x
            if self._tridim:
                m_ch  = tf.nn.avg_pool3d(x,  ksize=effective_window, strides=[1, 1, 1], padding='SAME')
                ms_ch = tf.nn.avg_pool3d(x2, ksize=effective_window, strides=[1, 1, 1], padding='SAME')
            else:
                m_ch  = tf.nn.avg_pool2d(x,  ksize=effective_window, strides=[1, 1], padding='SAME')
                ms_ch = tf.nn.avg_pool2d(x2, ksize=effective_window, strides=[1, 1], padding='SAME')

            # Reduce_mean over global axes (broadcast back via keepdims=True).
            global_axes = [1 + i for i in range(n_spatial) if is_global[i]]
            if global_axes:
                m_ch  = tf.reduce_mean(m_ch,  axis=global_axes, keepdims=True)
                ms_ch = tf.reduce_mean(ms_ch, axis=global_axes, keepdims=True)

            # Reshape pooled tensors to (..., G, Cg) (some spatial axes may be 1).
            m_ch_shape = tf.shape(m_ch)
            if self._tridim:
                stat_shape = tf.concat([m_ch_shape[:4], [self.groups, self._channels_per_group]], axis=0)
            else:
                stat_shape = tf.concat([m_ch_shape[:3], [self.groups, self._channels_per_group]], axis=0)
            m  = tf.reduce_mean(tf.reshape(m_ch,  stat_shape), axis=-1, keepdims=True)
            ms = tf.reduce_mean(tf.reshape(ms_ch, stat_shape), axis=-1, keepdims=True)
            var = tf.maximum(ms - m * m, tf.cast(0.0, m.dtype))
            x_g = tf.reshape(x, new_shape)
            inv = tf.math.rsqrt(var + tf.cast(self.epsilon, var.dtype))
            if self.scale:
                inv = inv * tf.reshape(self.gamma, self._vars_shape)
            res = -m * inv
            if self.center:
                res = res + tf.reshape(self.beta, self._vars_shape)
            x_g = x_g * inv + res
            out = tf.reshape(x_g, x_shape)
        return tf.cast(out, inputs.dtype)

