#! /usr/bin/env python


import os
import tensorflow as tf
import datasets
import cv2


batch_size = 10
epochs = None  # infinite epochs
destination_dirpath = '/Users/dave/Desktop/kitcar-workshop-destination/'
if not os.path.exists(destination_dirpath):
    os.makedirs(destination_dirpath)


def inference(input_tensor):
    tensor = tf.layers.conv2d(input_tensor, filters=3, kernel_size=(3, 3), padding='SAME')
    tensor = tf.layers.conv2d(tensor, filters=3, kernel_size=(3, 3), padding='SAME')
    return tensor


def loop():
    """This can be both inference or training call.
    """
    # Create all operations part of the graph
    iterator, feed_dict = datasets.dataset_images(batch_size=batch_size, epochs=epochs)
    # Important: Only this operation will yield images when executed.
    next_elem_op = iterator.get_next()

    # Create inference graph
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    output = inference(input_tensor=input_placeholder)

    # Define loss
    loss = tf.reduce_mean(tf.squared_difference(input_placeholder, output))

    # Create training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01, epsilon=0.1)
    optimization_op = optimizer.minimize(loss=loss)

    # Define summarization nodes
    summary_collection = 'single_collection'
    summary_loss = tf.summary.scalar('loss', tensor=loss, collections=(summary_collection))

    # Define what nodes of the graph we want to compute
    tensors_to_fetch = [output, optimization_op, summary_loss]  # What could be optimized here?  # TODO
    fetches = dict([(n.name, n) for n in tensors_to_fetch])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict=feed_dict)

        # Instantiate file writer
        file_writer = tf.summary.FileWriter(logdir=destination_dirpath)  # Important:  # TODO

        for step in range(1000):
            print('Step {}'.format(step))
            # batch_images = None
            # try:
            batch_images = sess.run(next_elem_op)
            # except tf.errors.OutOfRangeError:
            #     print('Iterated over {} epochs of dataset.'.format(epochs))
            #     exit()
            batch_images = sess.run(tf.cast(batch_images, tf.float32))  # otherwise uint8

            feed_dict = {input_placeholder: batch_images}
            evaluated_tensors = sess.run(fetches=fetches, feed_dict=feed_dict)

            if step % 20 == 0:
                print("Writing images for step {}".format(step))
                batch_output_images = evaluated_tensors[output.name]
                for i in range(len(batch_output_images)):
                    filename = "{}-{}.png".format(step, i)
                    output_img = cv2.cvtColor(batch_output_images[i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(destination_dirpath, filename), output_img)

                file_writer.add_summary(summary=evaluated_tensors[summary_loss.name],
                                        global_step=step)
                file_writer.flush()

            if step > 50:
                print('Finished.')
                exit()


loop()
