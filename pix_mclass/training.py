import numpy as np
from dataset_iterator import MultiChannelIterator
from dataset_iterator import extract_tile_random_zoom_function
from dataset_iterator.image_data_generator import data_generator_to_channel_postprocessing_fun
from .utils import ensure_multiplicity
def get_iterator(
    dataset, scaling_data_generator, illumination_data_generator=None,
    input_channel_keywords="raw", class_keyword:str="classes", train_group_keyword:str=None,
    batch_size:int=16, step_number:int=0,
    tiling_parameters:dict = None,
    elasticdeform_parameters=None,
    dtype="float32", shuffle:bool=True, memory_persistent:bool=False):

    """Training Iterator.

    Parameters
    ----------
    dataset : either string or DatasetIO object
        if a string: path to .h5 file containing
    train_group_keyword : str or list of stf
        keyword contained in the path of the training dataset, in order to exclude the evaluation dataset
    scaling_data_generator : ImageDataGenerator or list of ImageDataGenerator (one per channel)
        applied to each channel before elastic deform and tiling, only on channel image and not on classes.
        should only modify intensity and not deform the image as they will not be applied on class image
    illumination_data_generator : ImageDataGenerator or list of ImageDataGenerator (one per channel)
        applied to each channel after elastic deform and tiling, only on channel image and not on classes.
        should only modify intensity and not deform the image as they will not be applied on class image
    step_number : int

    Returns
    -------
    iterator of tiled images

    """

    extract_tiles_fun = None if tiling_parameters is None else extract_tile_random_zoom_function(**tiling_parameters)

    if isinstance(input_channel_keywords, (list, tuple)):
        channel_keywords = list(input_channel_keywords)
        channel_keywords.append(class_keyword)
        n_inputs = len(input_channel_keywords)
    else:
        channel_keywords = [input_channel_keywords, class_keyword]
        n_inputs = 1
    input_channels = list(range(n_inputs))
    scaling_data_generator = ensure_multiplicity(n_inputs, scaling_data_generator)
    if isinstance(scaling_data_generator, tuple):
        scaling_data_generator = [scaling_data_generator]
    scaling_data_generator.append(None) # classes
    if illumination_data_generator is not None:
        pp_fun = data_generator_to_channel_postprocessing_fun(illumination_data_generator, input_channels)
    else:
        pp_fun = None
    zero = np.zeros(shape=1, dtype=dtype)[0]
    one = np.ones(shape=1, dtype=dtype)[0]
    iterator_params = dict(dataset=dataset,
                           channel_keywords=channel_keywords,
                           input_channels = input_channels,
                           output_channels = [n_inputs],
                           mask_channels=[n_inputs],
                           group_keyword = train_group_keyword,
                           weight_map_functions = [lambda batch: np.where(batch==0, zero, one)],  # to avoid unlabeled data be mixed with label 0
                           output_postprocessing_functions = [lambda batch : np.where(batch>0, batch-one, zero)],  # labels must be in range [0, nlabels-1]
                           extract_tile_function = extract_tiles_fun,
                           image_data_generators = scaling_data_generator,
                           perform_data_augmentation=True,
                           elasticdeform_parameters=elasticdeform_parameters,
                           channels_postprocessing_function=pp_fun,
                           batch_size=batch_size,
                           incomplete_last_batch_mode="CONSTANT_SIZE",
                           shuffle=shuffle, step_number=step_number,
                           dtype=dtype, memory_persistent=memory_persistent)

    train_it = MultiChannelIterator(**iterator_params)
    return train_it
