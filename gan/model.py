import tensorflow as tf
from castanea.layers import conv2d, conv2d_transpose, linear, LayerParameter

def random_feature_generator(feature_size):
    with tf.name_scope('feature_generator'):
        feature = tf.truncated_normal(feature_size, dtype=tf.float32)
    
    return feature

def pseud_tanh(rate=0.01, lin_range=1.):
    return lambda x: tf.minimum(x*rate+lin_range,tf.maximum(x*rate-lin_range, x))

def upsample2x2(x,ks,ch,p):
    x = conv2d_transpose(x, ks, ks, ch, strides=[1,2,2,1], parameter=p)
    return x

def generator(feature, first_shape, num_upscale, reuse, training=False, var_device='/cpu:0'):
    x = feature
    ks = 5

    batch_normalize = True

    p0 = LayerParameter(
        with_batch_normalize=False,
        with_bias=False,
        rectifier=tf.tanh,
        training=training,
        var_device=var_device)
    p1 = LayerParameter(
        with_batch_normalize=batch_normalize,
        with_bias=False,
        rectifier=pseud_tanh(),
        training=training,
        var_device=var_device)
    p2 = LayerParameter(
        with_bias=False,
        with_batch_normalize=batch_normalize,
        rectifier=tf.nn.sigmoid,
        training=training,
        var_device=var_device)
    p3 = LayerParameter(
        with_batch_normalize=False,
        with_bias=False,
        rectifier=tf.nn.sigmoid,
        training=training,
        var_device=var_device)

    with tf.variable_scope('generator', reuse=reuse):
        xs = x.get_shape().as_list()
        x0 = x
        x = linear(x, first_shape, p0)

        ch = first_shape[3]
        print(x)

        for i in range(num_upscale):
            print(x)
            xs = x.get_shape().as_list()
            upsample_type = 2

            if upsample_type == 0:
                ch = ch // 2
                x = upsample2x2(x, ks, ch, p1)
            elif upsample_type == 1:
                ch = ch // 4
                x = tf.depth_to_space(x, 2)
                x = conv2d(x, ks, ks, ch, parameter=p1)
            elif upsample_type == 2:
                ch = ch // 2
                x = tf.image.resize_images(x, [xs[1]*2, xs[2]*2],
                    method=tf.image.ResizeMethod.BILINEAR)
                x = conv2d(x, ks, ks, ch, parameter=p1)
            x *= linear(x0, [xs[0],1,1,ch], p3)

        x = conv2d(x, ks, ks, 3, parameter=p2)

    return x

def discriminator(image, num_downscale, reuse, training=False, var_device='/cpu:0'):
    ks = 5
    p1 = LayerParameter(
        with_bias=False,
        with_batch_normalize=True,
        rectifier=pseud_tanh(),
        training=training,
        var_device=var_device)
    p2 = LayerParameter(
        with_bias=False,
        with_batch_normalize=True,
        rectifier=tf.nn.sigmoid,
        training=training,
        var_device=var_device)

    x = image - 0.5

    ch = 32

    with tf.variable_scope('discriminator', reuse=reuse):
        for i in range(num_downscale):
            x = conv2d(x, ks, ks, ch, strides=[1,2,2,1], parameter=p1)
            x = conv2d(x, ks, ks, ch, parameter=p1)
            ch = ch * 2

        x = linear(x, [-1, 1024], p2)
        x = linear(x, [-1, 1], p2)

        return x

def generator_loss(label_val, out):
    with tf.name_scope('generator_loss'):
        x = tf.squared_difference(label_val, out)
    return x

def discriminator_loss(label_val, out):
    with tf.name_scope('discriminator_loss'):
        x = tf.squared_difference(label_val, out)
    return x

