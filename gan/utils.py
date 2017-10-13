import tensorflow as tf
import castanea.layers

def read_image_op(filename_queue, reader, height, width):
    filename = filename_queue.dequeue()
    image_raw = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.to_float(image) / 255.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_images(image, [height, width])

    return image

