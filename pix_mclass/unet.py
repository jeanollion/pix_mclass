from tensorflow.keras.layers import Conv2D, Conv3D, Input, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
import keras.backend as K
import tensorflow as tf

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

def get_model(architecture_type:str, n_classes:int, n_inputs:int=1, n_input_channels:int=1, **kwargs):
    if architecture_type.lower() == "unet":
        filters = int(kwargs.get("filters", 256))
        n_downsampling = int(kwargs.get("n_downsampling", 4))
        maxpool = kwargs.get("maxpool", False)
        feature_settings = [
            {"filters": filters, "kernel_size": 5},
            {"filters": filters},
        ]
        decoder_settings = []
        encoder_settings = []
        for l in range(n_downsampling):
            current_filters = int(filters / 2**(min(n_downsampling - l, n_downsampling-1)))
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
        return get_unet(n_classes, n_inputs=n_inputs, n_input_channels=n_input_channels, encoder_settings=encoder_settings, feature_settings=feature_settings, decoder_settings=decoder_settings, skip_omit=kwargs.get("skip_omit", None))
    else:
        raise ValueError(f"Unknown architecture: {architecture_type}")

def get_unet(n_classes, n_inputs=1, n_input_channels=1, encoder_settings = ENCODER_SETTINGS, feature_settings= FEATURE_SETTINGS, decoder_settings=DECODER_SETTINGS, skip_sg=False, skip_omit=None, name = "unet", input_name = "unet_input"):
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
        down, residual, down_size = downsampling_block(input if d==0 else down, parameters, d, name)
        if d in skip_sg:
            residual = K.stop_gradient(residual)
        residuals.append(residual)
        downsample_size.append(down_size)
    up = downsampling_block(down, feature_settings, len(encoder_settings), name)
    for d in range(len(decoder_settings)-1, -1, -1):
        up=upsampling_block(up, None if d in skip_omit else residuals[d], decoder_settings[d], downsample_size[d], d, name)

    output = Conv2D(n_classes, kernel_size=3, padding='same', activation="softmax", name=f"{name}/output")(up)
    return Model(inputs, output, name=name)

def downsampling_block(input, parameters, l_idx, name):
    convolutions = [input]
    for i, params in enumerate(parameters):
        downsample = params.get("downscale", 1)
        maxpool = params.get("maxpool", False)
        assert downsample == 1 or i == len(parameters)-1, "downscale > 1 must be last convolution"
        conv = Conv2D(filters=params["filters"], kernel_size=params.get("kernel_size", 3), padding='same', activation=params.get("activation", "relu"), strides = 1 if maxpool else downsample, name=f"{name}/encoder/{l_idx}/conv_{i}")(convolutions[-1])
        convolutions.append(conv)
        if downsample>1 and maxpool:
            mp = MaxPool2D(pool_size = downsample, name=f"{name}/encoder/{l_idx}/mp_{i}")(convolutions[-1])
            convolutions.append(mp)
    downsample = parameters[-1].get("downscale", 1)
    if downsample>1:
        return convolutions[-1], convolutions[-2], downsample
    else:
        return convolutions[-1]

def upsampling_block(input, residual, params, stride, l_idx, name):
    up = Conv2DTranspose(params["filters"], kernel_size=params.get("up_kernel_size", 4), strides=stride, padding='same', activation=params.get("activation", "relu"), name=f"{name}/decoder/{l_idx}/upConv")(input)
    if residual is not None:
        concat = Concatenate(axis=-1, name = f"{name}/decoder/{l_idx}/concat")([residual, up])
    else:
        concat = up
    conv = Conv2D(filters=params["filters"], kernel_size=params.get("kernel_size", 3), padding='same', activation=params.get("activation", "relu"), name=f"{name}/decoder/{l_idx}/conv1")(concat)
    return Conv2D(filters=params["filters"] if l_idx==0 else params["filters"]//2, kernel_size=params.get("kernel_size", 3), padding='same', activation=params.get("activation", "relu"), name=f"{name}/decoder/{l_idx}/conv2")(conv)
