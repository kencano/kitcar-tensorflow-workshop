import os
import tensorflow as tf
import numpy as np
from glob import glob


BUFFER_SIZE = 4


def _finalise_dataset(dataset, batch_size, epochs):
    """Applies operations datasets generally have in common.
    :return: Iterator to dataset.
    """
    # Batch single elements together, drop odd last batch
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # Number of epochs to yield
    dataset = dataset.repeat(count=epochs)
    # Number of (until this point) preprocessed elements to keep in buffer
    dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
    iterator = dataset.make_initializable_iterator()
    return dataset, iterator


def dataset_range(batch_size, epochs):
    def _get_range(num_samples=25):
        for i in range(num_samples):
            yield np.float32(i)

    dataset = tf.data.Dataset.from_generator(generator=_get_range(), output_types=np.float32)
    _, iterator = _finalise_dataset(dataset, batch_size, epochs)
    return iterator


def dataset_random_tensors(batch_size, epochs):
    def _get_random_tensors(num_samples=25, image_shape=(96, 96, 3)):
        for _ in range(num_samples):
            yield np.random.rand(*image_shape)

    dataset = tf.data.Dataset.from_generator(generator=_get_random_tensors(), output_types=np.float32)
    _, iterator = _finalise_dataset(dataset, batch_size, epochs)
    return iterator


def dataset_images(batch_size, epochs):
    def _parse_function(filepath):
        file_content = tf.read_file(filepath)
        decoded_image = tf.image.decode_image(file_content)
        return decoded_image

    list_of_filepaths = glob('/Users/dave/Desktop/visualizing_gen_imgs_100k/AE_c16_d32-96_s6_I3x3_LR100_reg1e-6/*')
    filepath_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
    # Important: We base our dataset on a placeholder instead of filepaths, due to the potential amount of data
    dataset = tf.data.Dataset.from_tensor_slices(filepath_placeholder)
    # Read images
    dataset = dataset.map(_parse_function)
    # Batch single elements together, drop odd last batch
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # Number of epochs to yield
    dataset = dataset.repeat(count=epochs)
    # Number of (until this point) preprocessed elements to keep in buffer
    dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
    # Create iterator to dataset
    iterator = dataset.make_initializable_iterator()
    # feed_dict, in order to feed filepaths to the placeholder when initializing
    feed_dict = {filepath_placeholder: list_of_filepaths}
    return iterator, feed_dict
