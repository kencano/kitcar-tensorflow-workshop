#! /usr/bin/env python


import os
import tensorflow as tf
import datasets
import cv2


batch_size = 10
epochs = None  # infinite epochs
destination_dirpath = '/tmp/kitcar-workshop-destination/'
if not os.path.exists(destination_dirpath):
    os.makedirs(destination_dirpath)


def inference(input_tensor):
    # padding='SAME' will preserve spatial size
    tensor = tf.layers.conv2d(input_tensor, filters=3, kernel_size=(3, 3), padding='SAME')
    tensor = tf.layers.conv2d(tensor, filters=3, kernel_size=(3, 3), padding='SAME')
    return tensor


def loop():
    """This can be both inference or training call.
    """
    # Create all operations part of the graph
    iterator, iter_feed_dict = datasets.dataset_images(batch_size=batch_size, epochs=epochs)
    # Important: Only this operation will yield images when executed.
    next_elem_op = iterator.get_next()
    # We need to set the shape here manually, since the input pipeline cannot know what its own output shape will be a
    # priori, but that shape is required for most operations we want to append, e.g. conv layers.
    next_elem_op.set_shape((batch_size, None, None, 3))

    # Create inference graph
    # input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    output = inference(input_tensor=next_elem_op)

    # Define loss
    loss = tf.reduce_mean(tf.squared_difference(next_elem_op, output))

    # Create training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01, epsilon=0.1)
    optimization_op = optimizer.minimize(loss=loss)

    # Define summarization nodes
    summary_collection = 'single_collection'
    summary_loss = tf.summary.scalar('loss', tensor=loss, collections=(summary_collection,))

    # Define what nodes of the graph we want to compute
    tensors_to_fetch = [output, optimization_op, summary_loss]
    fetches = dict([(n.name, n) for n in tensors_to_fetch])

    with tf.Session() as sess:
        # We need to run the variable initializers for the network manually.
        sess.run(tf.global_variables_initializer())
        # Now that we are in a session, initialize the iterator
        sess.run(iterator.initializer, feed_dict=iter_feed_dict)

        # Instantiate file writer, which will write summaries to the disk
        file_writer = tf.summary.FileWriter(logdir=destination_dirpath)

        for step in range(1000):
            print('Step {}'.format(step))
            # Two alternatives to 1) Get next batch of images and 2) run optimization

            # Option 1: If inference graph was built upon a placeholder
            # batch_images = sess.run(next_elem_op)
            # loop_feed_dict = {input_placeholder: batch_images}
            # evaluated_tensors = sess.run(fetches=fetches, feed_dict=loop_feed_dict)

            # Better option 2: If the inference graph was built upon the iterator.get_next() tensor directly
            evaluated_tensors = sess.run(fetches=fetches)

            if step % 20 == 0:
                print("Writing images for step {}".format(step))
                batch_output_images = evaluated_tensors[output.name]
                for i in range(len(batch_output_images)):
                    filename = "{}-{}.png".format(step, i)
                    output_img = cv2.cvtColor(batch_output_images[i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(destination_dirpath, filename), output_img)

                # Add the evaluated summary values to the file_writer
                file_writer.add_summary(summary=evaluated_tensors[summary_loss.name],
                                        global_step=step)
                file_writer.flush()  # Force immediate write to disk

            if step > 200:
                print('Finished.')
                exit()


loop()
