import os
import tensorflow as tf
from glob import glob


BUFFER_SIZE = 4
FILEPATH_PATTERN = '~/Desktop/visualizing_gen_imgs_100k/AE_c16_d32-96_s6_I3x3_LR100_reg1e-6/*'


def dataset_images(batch_size, epochs, filepath_pattern=FILEPATH_PATTERN):

    def _parse_function(filepath):
        file_content = tf.read_file(filepath)
        decoded_image = tf.image.decode_image(file_content)
        return tf.cast(decoded_image, tf.float32)

    list_of_filepaths = glob(os.path.expanduser(filepath_pattern))
    filepath_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
    # Important: We base our dataset on a placeholder instead of filepaths, due to the potential amount of data
    # If we wouldn't do this, thefollowing Dataset operations, like reading/augmentation would apply to the entire
    # dataset instead of single filepaths/images
    dataset = tf.data.Dataset.from_tensor_slices(filepath_placeholder)
    # Shuffle Dataset
    dataset = dataset.shuffle(buffer_size=len(list_of_filepaths))
    # Map _parse_function across dataset; the dataset thereafter contains images instead of filepaths
    dataset = dataset.map(_parse_function)
    # Batch single elements together, drop odd last batch. The alternative would be `dataset.batch`, which would however
    # yield a batch with batch size smaller than defined when the dataset comes to an end.
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # Number of epochs to yield. An epoch is one iteration over the dataset.
    dataset = dataset.repeat(count=epochs)
    # Number of (until this point) preprocessed elements to keep in buffer
    # This single operation takes care of the input pipeline's queues
    dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
    # Create iterator to dataset which will have to be initialized inside a session, using the feed_dict below.
    iterator = dataset.make_initializable_iterator()
    # The usage of a placeholder requires us to feed values to it.
    # This feed_dict, will 'feed' filepaths to the placeholder when initializing, probably rather let the placeholder
    # point to the long list of filepaths
    feed_dict = {filepath_placeholder: list_of_filepaths}
    return iterator, feed_dict
