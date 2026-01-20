from keras.src.layers import Activation
from tensorflow.keras.layers import Conv2D, Conv3D, Input, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
import keras.backend as K
import tensorflow as tf

from dataset_iterator.keras_layers import Identity
from pix_mclass.layers import ScheduledDropout, ResidualGradientLimiter, ScheduledGradientWeight, \
    HybridThresholdL2Regularizer
from pix_mclass.window_spatial_attention import WindowSpatialAttention

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

def get_model(architecture_type:str, n_classes:int, n_inputs:int=1, n_input_channels:int=1, l2_reg:float=1e-4, batch_norm:bool=True, **kwargs):
    if architecture_type.lower() == "unet":
        filters = int(kwargs.get("filters", 256))
        min_filters = int(kwargs.get("filters_min", 32))
        n_downsampling = int(kwargs.get("n_downsampling", 4))
        attention_heads = int(kwargs.get("attention_heads",  0))
        attention_filters = int(kwargs.get("attention_filters", 64))
        attention_window_size = int(kwargs.get("attention_window", 16))
        attention  = attention_heads > 0 and attention_window_size > 0
        print(f"attention heads: {attention_heads} filters: {attention_filters} window size: {attention_window_size}")
        maxpool = kwargs.get("maxpool", False)
        feature_settings = [
            {"filters": filters, "kernel_size": 3 if attention else 5}
        ]
        if attention:
            feature_settings.append({"filters": attention_filters, "attention_heads": attention_heads, "window_size":attention_window_size})
        feature_settings.append({"filters": filters})

        decoder_settings = []
        encoder_settings = []
        for l in range(n_downsampling):
            current_filters = max(min_filters, int(filters / 2**(n_downsampling - l)))
            decoder_settings.append({"filters":current_filters})
            encoder_level = []
            if l > 1:
                encoder_level.append({"filters":current_filters, "kernel_size":5})
            encoder_level.append({"filters": current_filters})
            mul = 2 if l == n_downsampling-1 else 1
            encoder_level.append({"filters":current_filters * mul, "downscale":2, "maxpool":maxpool})
            encoder_settings.append(encoder_level)
        #print(f"encoder settings: {encoder_settings}")
        #print(f"feature settings: {feature_settings}")
        #print(f"decoder settings: {decoder_settings}")
        return get_unet(n_classes, n_inputs=n_inputs, n_input_channels=n_input_channels, encoder_settings=encoder_settings, feature_settings=feature_settings, decoder_settings=decoder_settings, skip_omit=kwargs.get("skip_omit", None), l2_reg=l2_reg, batch_norm=batch_norm)
    else:
        raise ValueError(f"Unknown architecture: {architecture_type}")

def get_unet(n_classes, n_inputs=1, n_input_channels=1, encoder_settings = ENCODER_SETTINGS, feature_settings= FEATURE_SETTINGS, decoder_settings=DECODER_SETTINGS, skip_sg=False, skip_omit=None, l2_reg:float=1e-4, batch_norm:bool=True, name = "unet", input_name = "unet_input"):
    assert len(encoder_settings)==len(decoder_settings), "encoder and decoder must have same depth"
    if n_inputs == 1:
        input = Input(shape = (None, None, n_input_channels), name=input_name)
        inputs = input
    else:
        if isinstance(input_name, str):
            input_name = [input_name+str(i) for i in range(n_inputs)]
        assert len(input_name) == n_inputs
        inputs = [Input(shape=(None, None, n_input_channels), name=input_name[i]) for i in range(n_inputs)]
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
        down, residual, down_size = encoder_block(input if d == 0 else down, parameters, d, len(encoder_settings), def_l2_reg=l2_reg, def_batch_norm=batch_norm)
        if d in skip_sg:
            residual = K.stop_gradient(residual)
        residuals.append(residual)
        downsample_size.append(down_size)
    up = encoder_block(down, feature_settings, len(encoder_settings), len(encoder_settings), def_l2_reg=l2_reg, def_batch_norm=batch_norm)
    for d in range(len(decoder_settings)-1, -1, -1):
        up=decoder_block(up, None if d in skip_omit else residuals[d], decoder_settings[d], downsample_size[d], d, def_l2_reg=l2_reg, def_batch_norm=batch_norm)

    output = Conv2D(n_classes, kernel_size=3, padding='same', dtype="float32", activation="softmax", name=f"output_conv")(up) # force float32 computation for softmax & loss precision
    return Model(inputs, output, name=name)

def encoder_block(input, parameters, l_idx, total_layers, def_l2_reg:float=1e-4, def_batch_norm:bool=True):
    x = input
    if parameters[-1].get("downscale", 1) > 1:
        res_idx = len(parameters) - 1 if parameters[-1].get("maxpool", False) else len(parameters) - 2
    else:
        res_idx = -1
    for i, params in enumerate(parameters):
        name = f"encoder{l_idx}_op{i}" if l_idx<total_layers else f"features_op{i}"
        downsample = params.get("downscale", 1)
        maxpool = params.get("maxpool", False)
        residual = i == res_idx
        batch_norm = params.get("batch_norm", def_batch_norm) and not residual
        assert downsample == 1 or i == len(parameters)-1, "downscale > 1 must be last convolution"
        l2_reg = params.get("l2_reg", def_l2_reg)
        if "attention_heads" in params:
            wsa = WindowSpatialAttention(num_heads=params["attention_heads"], attention_filters=params["filters"],
                                         l2_reg=l2_reg, position_encoding_l2_reg=l2_reg/10, add_distance_embedding=True,
                                         window_size=params["window_size"], layer_normalization=True, name=name)
            x = wsa(x)
        else:
            ker_reg, bias_reg = get_regularizers(l2_reg)
            x = Conv2D(filters=params["filters"], kernel_size=params.get("kernel_size", 3), padding='same',
                       kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                       activation=None if batch_norm else params.get("activation", "relu"), strides = 1 if maxpool else downsample,
                        name=name)(x)
            if batch_norm:
                x = BatchNormalization(name=name+"_bn")(x)
                x = Activation(params.get("activation", "relu"), name=name+"_act")(x)
        if residual:
            Id = tf.keras.layers.Identity if hasattr(tf.keras.layers, 'Identity') else Identity
            res = Id(name=f"encoder{l_idx}_skip", autocast=False)(x)
            progress_interval = [0.2, 0.7]
            min_progress = progress_interval[0] + (progress_interval[1] - progress_interval[0]) * (  total_layers - (l_idx + 1)) / total_layers
            max_progress = progress_interval[0] + (progress_interval[1] - progress_interval[0]) * (  total_layers - l_idx) / total_layers  # deepest skip reaches minimal rate before shallowest skip
            print(f"layer: {l_idx + 1}/{total_layers} progress range: [{min_progress}; {max_progress}]")
            res = ScheduledGradientWeight(min_progress=min_progress, max_progress=max_progress, name=f"res_grad_weight{l_idx}")(res)
            res, x = ResidualGradientLimiter(max_ratio=1., name=f"encoder{l_idx}_res_grad_limiter")( [res, x])
            res = ScheduledDropout(rate=0.0, max_rate=0.9, min_progress=min_progress, max_progress=max_progress, spatial=True, name=f"encoder{l_idx}_res_dropout")(res) # pushes the network to use deepest features
        if downsample>1 and maxpool:
            x = MaxPool2D(pool_size = downsample, name=f"encoder{l_idx}_mp")(x)
    downsample = parameters[-1].get("downscale", 1)
    if downsample>1:
        return x, res, downsample
    else:
        return x

def decoder_block(input, residual, params, stride, l_idx, def_l2_reg:float=1e-4, def_batch_norm:bool=True):
    ker_reg, bias_reg = get_regularizers(params.get("l2_reg", def_l2_reg))
    batch_norm = params.get("batch_norm", def_batch_norm)
    x = Conv2DTranspose(params["filters"], kernel_size=params.get("up_kernel_size", 4), strides=stride, padding='same',
                         kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                         activation=None if batch_norm else params.get("activation", "relu"), name=f"decoder{l_idx}_upConv")(input)
    if batch_norm:
        x = BatchNormalization(name=f"decoder{l_idx}_upConv_bn")(x)
        x = Activation(params.get("activation", "relu"), name=f"decoder{l_idx}_upConv_act")(x)
    if residual is not None:
        x = Concatenate(axis=-1, name = f"decoder{l_idx}_resConcat")([residual, x])

    x = Conv2D(filters=params["filters"], kernel_size=params.get("kernel_size", 3), padding='same',
                  kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                  activation=params.get("activation", "relu"), name=f"decoder{l_idx}_op1")(x)
    x = Conv2D(filters=params["filters"], kernel_size=params.get("kernel_size", 3),
                  kernel_regularizer=ker_reg, bias_regularizer=bias_reg,
                  padding='same', activation=None if batch_norm else params.get("activation", "relu"), name=f"decoder{l_idx}_op2")(x)
    if batch_norm:
        x = BatchNormalization(name=f"decoder{l_idx}_op2_bn")(x)
        x = Activation(params.get("activation", "relu"), name=f"decoder{l_idx}_op2_act")(x)
    return x

def get_regularizers(l2_reg:float):
    ker_reg = HybridThresholdL2Regularizer(elementwise_strength=l2_reg, directional_strength=l2_reg * 10) if l2_reg > 0 else None
    bias_reg = HybridThresholdL2Regularizer(elementwise_strength=l2_reg, directional_strength=0) if l2_reg > 0 else None
    return ker_reg, bias_reg