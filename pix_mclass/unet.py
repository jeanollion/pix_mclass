import sys

from tensorflow.keras.layers import (
    Conv2D, Conv3D,
    Input,
    MaxPool2D, MaxPool3D,
    Conv2DTranspose, Conv3DTranspose,
    Concatenate, BatchNormalization, Activation,
)
from tensorflow.keras.models import Model
import keras.backend as K
import tensorflow as tf

from dataset_iterator.utils import ensure_multiplicity
from pix_mclass.layers import ResidualGradientLimiter, HybridThresholdL2Regularizer
from pix_mclass.window_group_norm import WindowGroupNormalization
from pix_mclass.activations import get_activation

ENCODER_SETTINGS = [
    [ # l1 = 128 -> 64
        {"filters":32},
        {"filters":32, "downscale":2, "maxpool":False}
    ],
    [  # l2 64 -> 32
        {"filters":32},
        {"filters":32, "downscale":2,"maxpool":False}
    ],
    [ # l3: 32->16
        {"filters":64, "kernel_size":5},
        {"filters":64},
        {"filters":64, "downscale":2,"maxpool":False},
    ],
    [ # l4: 16 -> 8
        {"filters":128, "kernel_size":5},
        {"filters":128},
        {"filters":256, "downscale":2,"maxpool":False}
    ]
]
FEATURE_SETTINGS = [
    {"filters":256, "kernel_size":5},
    {"filters":256},
]

DECODER_SETTINGS = [
    {"filters":32},
    {"filters":32},
    {"filters":64},
    {"filters":128}
]


def _get_layers(tridimensional_mode:bool):
    """Return (Conv, ConvTranspose, MaxPool) classes for the requested dimensionality."""
    if tridimensional_mode:
        return Conv3D, Conv3DTranspose, MaxPool3D
    return Conv2D, Conv2DTranspose, MaxPool2D


_NO_NORM = (None, False, "no_norm", "none")
_BATCH_NORM = ("batch_norm", "batch", "bn")
_WGN = ("wgn", "window_group", "window_group_norm")


def _canon_norm(normalization):
    """Lower-case string normalization specs; pass through None/False unchanged."""
    return normalization.lower() if isinstance(normalization, str) else normalization


def _is_wgn(normalization):
    return _canon_norm(normalization) in _WGN


def _norm_enabled(normalization):
    """True when ``normalization`` requests an actual layer (not no_norm/None)."""
    return _canon_norm(normalization) not in _NO_NORM


def _make_norm(normalization:str, name:str, norm_window_size=None):
    """Build a normalization layer from a ``normalization`` spec.

    ``normalization`` is one of:
      - ``"no_norm"`` / ``None`` / ``False`` -> no layer (returns ``None``)
      - ``"batch_norm"``                      -> ``BatchNormalization``
      - ``"wgn"``                             -> ``WindowGroupNormalization``

    ``WindowGroupNormalization`` (wgn) computes its statistics from the input
    itself (locally pooled, size-invariant) so — unlike BatchNormalization — it
    behaves identically at train and inference time (no moving-average
    population stats), which avoids the train/inference collapse to a uniform
    output that BN can cause in deep 3D nets. ``norm_window_size`` (int or
    per-axis tuple, e.g. ``(1, 32, 32)`` for per-Z-slice stats on anisotropic 3D
    data) is forwarded to WindowGroupNormalization; ``None`` keeps its default.
    """
    normalization = _canon_norm(normalization)
    if normalization in _NO_NORM:
        return None
    if normalization in _WGN:
        kw = {} if norm_window_size is None else {"window_size": norm_window_size}
        return WindowGroupNormalization(name=name, **kw)
    if normalization in _BATCH_NORM:
        return BatchNormalization(name=name)
    raise ValueError(
        f"Unknown normalization {normalization!r}; expected one of "
        f"'no_norm', 'batch_norm', 'wgn' (or None)."
    )


def _is_downsampling(downscale):
    """True if at least one axis is being downsampled (downscale > 1)."""
    if isinstance(downscale, (list, tuple)):
        return any(d > 1 for d in downscale)
    return downscale > 1


def _resolve_level_downscale(current_shape, n_spatial_dims:int, default:int=2):
    """Compute the downscale value for one encoder level, capped per-axis.

    ``current_shape`` is a mutable list of length ``n_spatial_dims`` whose
    entries are either an int (constrained axis size) or ``None`` (unconstrained).
    For each constrained axis whose size is below ``default`` the per-axis
    downscale is forced to 1; otherwise it is ``default`` and the axis size is
    floor-divided in place. Returns ``default`` (int) when all axes downsample
    by ``default``, otherwise a tuple of length ``n_spatial_dims``.
    """
    if current_shape is None:
        return default
    ds = []
    for ax in range(n_spatial_dims):
        sz = current_shape[ax]
        if sz is None:
            ds.append(default)
        elif sz >= default:
            ds.append(default)
            current_shape[ax] = sz // default
        else:
            ds.append(1)
    if all(d == default for d in ds):
        return default
    return tuple(ds)


def _normalize_input_shape(input_shape, n_spatial_dims:int):
    """Return a mutable per-axis size list (None for unconstrained), or None.

    ``0`` and ``None`` entries are treated as unconstrained.
    """
    if input_shape is None:
        return None
    if isinstance(input_shape, int):
        input_shape = (input_shape,) * n_spatial_dims
    assert len(input_shape) == n_spatial_dims, (
        f"input_shape must have length {n_spatial_dims} (got {len(input_shape)})"
    )
    return [None if (s is None or s == 0) else int(s) for s in input_shape]


def _resolve_kernel_size(target_kernel, current_shape, n_spatial_dims:int):
    """Cap a per-axis kernel size to fit ``current_shape``.

    A kernel ``k`` (with dilation 1) is allowed on an axis of size ``dim``
    only when ``dim >= 2 * (k - 1)``. When this fails, the kernel is reduced
    to the next-lower odd integer (``k -= 2``), with a floor of 1. Even
    target kernels are first decremented by 1 to become odd. Axes that are
    unconstrained (``None`` size) keep the requested value. Returns an int
    when all axes resolve to the same kernel, otherwise a tuple.
    """
    if current_shape is None:
        return target_kernel
    if isinstance(target_kernel, (list, tuple)):
        assert len(target_kernel) == n_spatial_dims, (
            f"kernel_size length must be {n_spatial_dims} (got {len(target_kernel)})"
        )
        per_axis = [int(k) for k in target_kernel]
    else:
        per_axis = [int(target_kernel)] * n_spatial_dims
    for ax in range(n_spatial_dims):
        sz = current_shape[ax]
        if sz is None or sz <= 0:
            continue
        ker = per_axis[ax]
        if ker > 1 and ker % 2 == 0:
            ker -= 1  # only odd kernels are allowed
        while ker > 1 and sz < 2 * (ker - 1):
            ker -= 2
        per_axis[ax] = max(1, ker)
    if all(k == per_axis[0] for k in per_axis):
        return per_axis[0]
    return tuple(per_axis)


def _ds_per_axis(downscale, n_spatial_dims:int):
    """Expand a downscale value (int or per-axis tuple) to a per-axis int list."""
    if isinstance(downscale, (list, tuple)):
        return [int(d) for d in downscale]
    return [int(downscale)] * n_spatial_dims


def _window_per_axis(window_size, n_spatial_dims:int):
    """Expand a window_size (int or per-axis tuple) to a per-axis int list, or None."""
    if window_size is None:
        return None
    if isinstance(window_size, (list, tuple)):
        assert len(window_size) == n_spatial_dims, (
            f"norm_window_size length must be {n_spatial_dims} (got {len(window_size)})"
        )
        return [int(w) for w in window_size]
    return [int(window_size)] * n_spatial_dims


def get_model(architecture_type:str, n_classes:int, n_inputs:int=1, n_input_channels:int=1, l2_reg:float=1e-4, normalization:str=None, activation:str="dscrelu32", tridimensional_mode:bool=False, input_shape=None, norm_window_size=None, norm_window_scaling:bool=True, normalization_scope:str="all", **kwargs):
    """Build a segmentation model.

    Parameters
    ----------
    input_shape : int, tuple/list of int, or None
        Optional spatial shape (no batch / no channels) used **only** to cap
        the per-axis downscale of each encoder level. Entries set to ``None``
        or ``0`` are treated as unconstrained. Length must match
        ``n_spatial_dims`` (2 by default, 3 when ``tridimensional_mode``).
    normalization : {"no_norm", "batch_norm", "wgn"} or None
        Normalization layer used after convolutions. ``None`` -> ``"wgn"`` in 3D,
        ``"batch_norm"`` in 2D. ``"wgn"`` (WindowGroupNormalization) is recommended
        for 3D because it is statistics-stable between train and inference (no
        moving-average population stats), avoiding the uniform-output collapse that
        BatchNormalization can cause in deep 3D nets. The legacy boolean
        ``batch_norm`` kwarg is still accepted (``True`` -> ``"batch_norm"``,
        ``False`` -> ``"no_norm"``) but ``normalization`` takes precedence.
    norm_window_size : int, tuple/list or None
        Spatial window for WindowGroupNormalization, interpreted as the window at
        the **bottleneck** (deepest / lowest-resolution) level. ``None`` ->
        defaults to the spatial size at the feature/bottleneck level (derived from
        ``input_shape``), i.e. a global window at the bottleneck that, with
        ``norm_window_scaling`` on, stays global at every level. Requires
        ``input_shape`` to be (fully) provided; if it is not and ``norm_window_size``
        is also ``None``, a ``ValueError`` is raised (the window cannot be derived).
    norm_window_scaling : bool
        When ``True`` (default), the WGN window is scaled up at the shallower
        encoder/decoder levels by the per-axis cumulative downscale factor between
        that level and the bottleneck, so every level normalizes over the same
        physical receptive region (``norm_window_size`` being the size at the
        bottleneck). When ``False``, the same ``norm_window_size`` is used verbatim
        at every level (legacy behavior).
    normalization_scope : {"all", "per_block", "resample", "none"}
        Which convolutions get a normalization layer:
          - ``"all"`` (default): every norm-eligible conv (residual ops are always
            skipped to keep the skip path clean). Matches the nnU-Net recipe.
          - ``"per_block"``: a single norm per block, at the block *entry* (the first
            non-residual conv; the decoder up-conv when present). ConvNeXt-style.
          - ``"resample"``: a single norm per block, at the resolution *transition*
            (the encoder conv that feeds the down-sampling; the decoder up-conv).
          - ``"none"``: no normalization anywhere.
    """
    # Backward compatibility: honor the legacy boolean ``batch_norm`` kwarg when
    # ``normalization`` was not given explicitly.
    legacy_batch_norm = kwargs.pop("batch_norm", None)
    if normalization is None and legacy_batch_norm is not None:
        normalization = "batch_norm" if legacy_batch_norm else "no_norm"
    if normalization == "no_norm":
        normalization_scope = "none"
    # Default normalization: WindowGroupNorm in 3D (train/inference stable), BatchNorm in 2D.
    if normalization is None:
        normalization = "wgn" if tridimensional_mode else "batch_norm"
    normalization = _canon_norm(normalization)
    print(f"normalization: {normalization} scope: {normalization_scope}, activation: {activation}", flush=True)
    if architecture_type.lower() == "unet":
        filters = int(kwargs.get("filters", 256))
        min_filters = int(kwargs.get("filters_min", 32))
        n_downsampling = int(kwargs.get("n_downsampling", 4))
        maxpool = kwargs.get("maxpool", False)
        n_spatial_dims = 3 if tridimensional_mode else 2
        current_shape = _normalize_input_shape(input_shape, n_spatial_dims)

        # Snapshot of the spatial shape at the input of each encoder level
        # (= the output shape of the matching decoder level).
        shapes_per_level = []
        level_downscales = []
        decoder_settings = []
        encoder_settings = []
        for l in range(n_downsampling):
            shapes_per_level.append(list(current_shape) if current_shape is not None else None)
            current_filters = max(min_filters, int(filters / 2**(n_downsampling - l)))
            ker3 = _resolve_kernel_size(3, current_shape, n_spatial_dims)
            ker5 = _resolve_kernel_size(5, current_shape, n_spatial_dims)
            encoder_level = []
            if l > 1:
                encoder_level.append({"filters":current_filters, "kernel_size":ker5})
            encoder_level.append({"filters": current_filters, "kernel_size":ker3})
            mul = 2 if l == n_downsampling-1 else 1
            level_downscale = _resolve_level_downscale(current_shape, n_spatial_dims, default=2)
            level_downscales.append(level_downscale)
            encoder_level.append({"filters":current_filters * mul, "kernel_size":ker3, "downscale":level_downscale, "maxpool":maxpool})
            encoder_settings.append(encoder_level)

        # Bottleneck at the deepest spatial shape
        ker5_btl = _resolve_kernel_size(5, current_shape, n_spatial_dims)
        ker3_btl = _resolve_kernel_size(3, current_shape, n_spatial_dims)
        feature_settings = [
            {"filters": filters, "kernel_size": ker5_btl},
            {"filters": filters, "kernel_size": ker3_btl},
        ]
        print("shape at feature level: ", current_shape)
        # Default WGN window = the spatial size at the feature/bottleneck level.
        # This needs a (fully) known feature-level shape, i.e. a provided input_shape;
        # otherwise the window cannot be derived and must be given explicitly.
        if _is_wgn(normalization) and norm_window_size is None:
            if current_shape is None or any(s is None for s in current_shape):
                raise ValueError(
                    "norm_window_size must be provided when input_shape is not (fully) provided: "
                    "with WindowGroupNormalization the default window is derived from the "
                    "feature-level (bottleneck) spatial size, which requires a known input_shape."
                )
            norm_window_size = tuple(int(s) for s in current_shape)
        # Decoder convs run at the resolution of their matching encoder level input
        for l in range(n_downsampling):
            shape_l = shapes_per_level[l]
            current_filters = max(min_filters, int(filters / 2**(n_downsampling - l)))
            decoder_settings.append({
                "filters": current_filters,
                "kernel_size": _resolve_kernel_size(3, shape_l, n_spatial_dims),
            })

        # Scale the WGN window per level: norm_window_size is the window at the
        # bottleneck; at each shallower level it is multiplied by the per-axis
        # cumulative downscale factor between that level and the bottleneck, so the
        # normalized region covers the same physical extent at every resolution.
        base_ws = _window_per_axis(norm_window_size, n_spatial_dims)
        if norm_window_scaling and base_ws is not None:
            for l in range(n_downsampling):
                factor = [1] * n_spatial_dims
                for k in range(l, n_downsampling):  # downscales from level l down to bottleneck
                    ds = _ds_per_axis(level_downscales[k], n_spatial_dims)
                    factor = [factor[a] * ds[a] for a in range(n_spatial_dims)]
                window_l = tuple(base_ws[a] * factor[a] for a in range(n_spatial_dims))
                for op in encoder_settings[l]:
                    op["norm_window_size"] = window_l
                decoder_settings[l]["norm_window_size"] = window_l
            # bottleneck (feature_settings) keeps the unscaled base window (factor 1),
            # supplied via the norm_window_size argument forwarded to get_unet.

        # Output Conv runs at the input resolution
        output_shape = shapes_per_level[0] if shapes_per_level else current_shape
        output_kernel_size = _resolve_kernel_size(3, output_shape, n_spatial_dims)

        return get_unet(
            n_classes,
            n_inputs=n_inputs, n_input_channels=n_input_channels,
            encoder_settings=encoder_settings, feature_settings=feature_settings,
            decoder_settings=decoder_settings,
            skip_omit=kwargs.get("skip_omit", None),
            l2_reg=l2_reg, activation=activation,
            normalization=normalization, norm_window_size=norm_window_size,
            normalization_scope=normalization_scope,
            tridimensional_mode=tridimensional_mode,
            output_kernel_size=output_kernel_size,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture_type}")

def get_unet(n_classes, n_inputs=1, n_input_channels=1, encoder_settings = ENCODER_SETTINGS, feature_settings= FEATURE_SETTINGS, decoder_settings=DECODER_SETTINGS, skip_sg=False, skip_omit=None, l2_reg:float=1e-4, activation="dscrelu32", normalization:str="batch_norm", norm_window_size=None, normalization_scope:str="all", tridimensional_mode:bool=False, output_kernel_size=3, name = "unet", input_name = "unet_input"):
    assert len(encoder_settings)==len(decoder_settings), "encoder and decoder must have same depth"
    n_spatial_dims = 3 if tridimensional_mode else 2
    keras_input_shape = (None,) * n_spatial_dims + (n_input_channels,)
    Conv, _, _ = _get_layers(tridimensional_mode)
    if n_inputs == 1:
        input = Input(shape=keras_input_shape, name=input_name)
        inputs = input
    else:
        if isinstance(input_name, str):
            input_name = [input_name+str(i) for i in range(n_inputs)]
        assert len(input_name) == n_inputs
        inputs = [Input(shape=keras_input_shape, name=input_name[i]) for i in range(n_inputs)]
        input = Concatenate(axis=-1)(inputs)
    if skip_sg==True:
        skip_sg = [i for i in range(len(ENCODER_SETTINGS))]
    elif skip_sg==False or skip_sg is None:
        skip_sg = []
    elif isinstance(skip_sg, int):
        skip_sg = [skip_sg]
    assert isinstance(skip_sg, (list, tuple)), "invalid argument: skip_sg should be either bool, None, int or tuple/list of int"
    if skip_omit is None:
        skip_omit = []
    elif isinstance(skip_omit, int):
        skip_omit = [skip_omit]
    assert isinstance(skip_omit, (list, tuple)), "invalid argument: skip_omit should be either None, int or tuple/list of int"
    residuals = []
    downsample_size = []
    for d, parameters in enumerate(encoder_settings):
        down, residual, down_size = encoder_block(input if d == 0 else down, parameters, d, len(encoder_settings), def_l2_reg=l2_reg, def_activation=activation, def_normalization=normalization, norm_window_size=norm_window_size, def_normalization_scope=normalization_scope, tridimensional_mode=tridimensional_mode)
        if d in skip_sg:
            residual = K.stop_gradient(residual)
        residuals.append(residual)
        downsample_size.append(down_size)
    up = encoder_block(down, feature_settings, len(encoder_settings), len(encoder_settings), def_l2_reg=l2_reg, def_activation=activation, def_normalization=normalization, norm_window_size=norm_window_size, def_normalization_scope=normalization_scope, tridimensional_mode=tridimensional_mode)
    for d in range(len(decoder_settings)-1, -1, -1):
        up=decoder_block(up, None if d in skip_omit else residuals[d], decoder_settings[d], downsample_size[d], d, def_l2_reg=l2_reg, def_activation=activation, def_normalization=normalization, norm_window_size=norm_window_size, def_normalization_scope=normalization_scope, tridimensional_mode=tridimensional_mode)

    output = Conv(n_classes, kernel_size=output_kernel_size, padding='same', dtype="float32", activation="softmax", name=f"output_conv")(up) # force float32 computation for softmax & loss precision
    return Model(inputs, output, name=name)

def encoder_block(input, parameters, l_idx, total_layers, def_l2_reg:float=1e-4, def_activation="dscrelu32", def_normalization:str="batch_norm", norm_window_size=None, def_normalization_scope:str="all", tridimensional_mode:bool=False):
    Conv, _, MaxPool = _get_layers(tridimensional_mode)
    x = input
    if _is_downsampling(parameters[-1].get("downscale", 1)):
        res_idx = len(parameters) - 1 if parameters[-1].get("maxpool", False) else len(parameters) - 2
    else:
        res_idx = -1
    # Select which (non-residual) convs get a norm layer, per normalization_scope.
    non_res = [i for i in range(len(parameters)) if i != res_idx]
    if def_normalization_scope in ("none", "no_norm"):  # no normalization anywhere
        norm_idx = set()
    elif def_normalization_scope == "per_block":    # one norm, at block entry (first conv)
        norm_idx = {non_res[0]} if non_res else set()
    elif def_normalization_scope == "resample":     # one norm, at the conv feeding the down-sample
        norm_idx = {non_res[-1]} if non_res else set()
    else:                                           # "all"
        norm_idx = set(non_res)
    for i, params in enumerate(parameters):
        name = f"encoder{l_idx}_op{i}" if l_idx<total_layers else f"features_op{i}"
        downsample = params.get("downscale", 1)
        maxpool = params.get("maxpool", False)
        residual = i == res_idx
        # residual ops are never normalized; otherwise build the configured norm layer
        # only at the conv(s) selected by normalization_scope.
        norm_layer = _make_norm(params.get("normalization", def_normalization), name+"_norm", params.get("norm_window_size", norm_window_size)) if i in norm_idx else None
        use_norm = norm_layer is not None
        activation = get_activation(params.get("activation", def_activation))
        assert not _is_downsampling(downsample) or i == len(parameters)-1, "downscale > 1 must be on last convolution"
        l2_reg = params.get("l2_reg", def_l2_reg)
        ker_reg, bias_reg = get_regularizers(l2_reg)
        x = Conv(filters=params["filters"], kernel_size=params.get("kernel_size", 3), padding='same',
                   kernel_initializer=get_kernel_initializer(activation),
                   kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                   activation=None if use_norm else activation, strides = 1 if maxpool else downsample,
                   name=name)(x)
        if use_norm:
            x = norm_layer(x)
            x = Activation(activation, name=name+"_act")(x)
        if residual:
            res, x = ResidualGradientLimiter(max_ratio=1., name=f"encoder{l_idx}_res_grad_limiter")( x )
        if _is_downsampling(downsample) and maxpool:
            x = MaxPool(pool_size = downsample, name=f"encoder{l_idx}_mp")(x)
    downsample = parameters[-1].get("downscale", 1)
    if _is_downsampling(downsample):
        return x, res, downsample
    else:
        return x

def decoder_block(input, residual, params, stride, l_idx, def_l2_reg:float=1e-4, def_activation="dscrelu32", def_normalization:str="batch_norm", norm_window_size=None, def_normalization_scope:str="all", tridimensional_mode:bool=False):
    Conv, ConvTranspose, _ = _get_layers(tridimensional_mode)
    ker_reg, bias_reg = get_regularizers(params.get("l2_reg", def_l2_reg))
    normalization = params.get("normalization", def_normalization)
    norm_on = _norm_enabled(normalization)
    scope = params.get("normalization_scope", def_normalization_scope)
    win = params.get("norm_window_size", norm_window_size)
    activation =  get_activation(params.get("activation", def_activation))
    stride = ensure_multiplicity(3 if tridimensional_mode else 2, stride)
    has_up = any(s != 1 for s in stride)
    # Which decoder convs get a norm, per normalization_scope. The up-conv is the
    # resolution transition; op2 is the block's output conv. (op1 is never normalized.)
    if scope in ("none", "no_norm"):                # no normalization anywhere
        norm_up = False
        norm_op2 = False
    else:
        norm_up = norm_on and has_up
        if scope == "all":
            norm_op2 = norm_on
        elif scope == "per_block":
            norm_op2 = norm_on and not has_up   # one norm: up-conv if present, else op2
        else:  # "resample": norm only the up-conv (no up-conv -> no norm in this block)
            norm_op2 = False
    if has_up:
        x = ConvTranspose(params["filters"], kernel_size=[s * 2 for s in stride], strides=stride, padding='same',
                             kernel_initializer=get_kernel_initializer(activation),
                             kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                             activation=None if norm_up else activation, name=f"decoder{l_idx}_upConv")(input)
        if norm_up:
            x = _make_norm(normalization, f"decoder{l_idx}_upConv_bn", win)(x)
            x = Activation(activation, name=f"decoder{l_idx}_upConv_act")(x)
    else:
        x = input

    if residual is not None:
        x = Concatenate(axis=-1, name = f"decoder{l_idx}_resConcat")([residual, x])

    x = Conv(filters=params["filters"], kernel_size=params.get("kernel_size", 3), padding='same',
                  kernel_initializer = get_kernel_initializer(activation),
                  kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                  activation=activation, name=f"decoder{l_idx}_op1")(x)
    x = Conv(filters=params["filters"], kernel_size=params.get("kernel_size", 3),
                  kernel_initializer=get_kernel_initializer(activation),
                  kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                  padding='same', activation=None if norm_op2 else activation, name=f"decoder{l_idx}_op2")(x)
    if norm_op2:
        x = _make_norm(normalization, f"decoder{l_idx}_op2_bn", win)(x)
        x = Activation(activation, name=f"decoder{l_idx}_op2_act")(x)
    return x

def get_regularizers(l2_reg:float):
    ker_reg = HybridThresholdL2Regularizer(elementwise_strength=l2_reg, directional_strength=l2_reg * 10) if l2_reg > 0 else None
    bias_reg = HybridThresholdL2Regularizer(elementwise_strength=l2_reg, directional_strength=0) if l2_reg > 0 else None
    return ker_reg, bias_reg

def get_kernel_initializer(activation:str):
    return "glorot_uniform"
