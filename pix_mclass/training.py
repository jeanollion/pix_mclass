import numpy as np
from dataset_iterator import MultiChannelIterator, PreProcessingImageGenerator
from dataset_iterator import extract_tile_function, extract_tile_random_zoom_function
from math import ceil

def get_iterator(
    dataset, scaling_function,
    input_channel_keywords:str="raw", class_keyword:str="classes", train_group_keyword:str=None,
    batch_size:int=16,min_step_number:int=200,
    patch_shape:tuple=(256,256), n_tiles:int=8, zoom_range=[0.8,1.2],aspect_ratio_range=[0.8,1.2],
    dtype="float32" ):

    """Training Iterator.

    Parameters
    ----------
    dataset : either string or DatasetIO object
        if a string: path to .h5 file containing
    train_group_keyword : string
        keyword contained in the path of the training dataset, in order to exclude the evaluation dataset
    scaling_function : callable
        callable applied to each image before tiling
    patch_shape : tuple. If None: no tiling is applied
    n_tiles : int
    min_step_number : int

    Returns
    -------
    iterator of tiled images

    """

    assert callable(scaling_function)
    pp_fun = PreProcessingImageGenerator(scaling_function)
    extract_tiles_fun = None if patch_shape is None else extract_tile_random_zoom_function(patch_shape, n_tiles=n_tiles, zoom_range=zoom_range, aspect_ratio_range=aspect_ratio_range, perform_augmentation=True, augmentation_rotate=True, random_stride=True )

    if isinstance(input_channel_keywords, (list, tuple)):
        channel_keywords = list(input_channel_keywords)
        channel_keywords.append(class_keyword)
        n_inputs = len(input_channel_keywords)
    else:
        channel_keywords = [input_channel_keywords, class_keyword]
        input_channel_keywords = [input_channel_keywords]
        n_inputs = 1
    image_data_generators = [pp_fun]*n_inputs
    image_data_generators.append(None)
    zero = np.zeros(shape=1, dtype=dtype)
    one = np.ones(shape=1, dtype=dtype)
    iterator_params = dict(dataset=dataset,
        channel_keywords=channel_keywords,
        input_channels = list(range(n_inputs)),
        output_channels = [n_inputs],
        mask_channels=[n_inputs],
        group_keyword = train_group_keyword,
        weight_map_functions = [lambda batch: np.where(batch==0, zero, one)], # to avoid unlabeled data be mixed with label 0
        output_postprocessing_functions = [lambda batch : np.where(batch>0, batch-one, zero)], # labels must be in range [0, nlabels-1]
        extract_tile_function = extract_tiles_fun,
        image_data_generators = image_data_generators,
        perform_data_augmentation=True,
        batch_size=batch_size,
        incomplete_last_batch_mode="CONSTANT_SIZE",
        shuffle=True,
        dtype=dtype)

    train_it = MultiChannelIterator(**iterator_params)
    train_idxs = train_it.allowed_indexes
    if min_step_number is not None: # if min_step_number > iterator length -> need to repeat some batch
        rep = ceil(train_it.batch_size * min_step_number / len(train_idxs) )
        if rep>1:
            train_idxs = np.array(list(train_idxs) * rep)
            train_it.set_allowed_indexes(train_idxs)
    return train_it
