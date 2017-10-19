from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

import gan.utils as utils
import gan.model as model

DEFAULT_NUM_ITERATIONS=100000
DEFAULT_MINIBATCH_SIZE=32
DEFAULT_LOGDIR='./logs'
DEFAULT_INPUT_MODEL='model_training'
DEFAULT_OUTPUT_MODEL='model_training'

DEFAULT_INPUT_THREADS=8
DEFAULT_INPUT_QUEUE_MIN=2000
DEFAULT_INPUT_QUEUE_MAX=10000

NUM_DEVICES=2
VAR_DEVICE='/gpu:0'

DEFAULT_INPUT_WIDTH=64
DEFAULT_INPUT_HEIGHT=64
DEFAULT_FEATURE_DIM=128
FEATURE_SHAPE = [DEFAULT_FEATURE_DIM]
FIRST_SHAPE = [-1, 4, 4, 1024]
IMAGE_LIST_FILE = './celeba_samples.txt'
NUM_UPSAMPLE=4
NUM_DOWNSAMPLE=4
D_THRESH = 0.5
LEARNING_RATE_G = 1e-3
LEARNING_RATE_D = 1e-3
USE_ADAM=False

def create_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-n','--num-iterations', type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument('-m','--minibatch-size', type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument('-l','--logdir', type=str, default=DEFAULT_LOGDIR)
    parser.add_argument('-i','--input-model', type=str, default=DEFAULT_INPUT_MODEL)
    parser.add_argument('-o','--output-model', type=str, default=DEFAULT_OUTPUT_MODEL)

    parser.add_argument('--input-threads', type=int, default=DEFAULT_INPUT_THREADS)
    parser.add_argument('--input-queue_min', type=int, default=DEFAULT_INPUT_QUEUE_MIN)
    parser.add_argument('--input-queue-max', type=int, default=DEFAULT_INPUT_QUEUE_MAX)

    return parser

def add_application_arguments(parser):
    parser.add_argument('--width', type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_INPUT_HEIGHT)

    return parser

def main():
    parser = create_argument_parser()
    parser = add_application_arguments(parser)
    args = parser.parse_args()
    proc(args)

def average_gradients(tower_grads):
    average_grads = []

    with tf.name_scope('average_gradients'):
        for grad_and_vars in zip(*tower_grads):
            grads = []

            for g, u in grad_and_vars:
                expanded_g = tf.expand_dims(g,0)
                grads.append(expanded_g)
                tf.summary.histogram(u.name, g, collections=['histogram', 'grads'])

            grad = tf.concat(axis=0,values=grads)
            grad = tf.reduce_mean(grad,0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

def proc(args):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.variable_scope('global'):
            global_step = tf.get_variable(
                'global_step', shape=[],
                initializer=tf.constant_initializer(0, dtype=tf.int64), trainable=False)
            d_train_count = tf.get_variable(
                'd_train_count', shape=[],
                initializer=tf.constant_initializer(0, dtype=tf.int64), trainable=False)

        with tf.name_scope('input'):
            filenames = open(IMAGE_LIST_FILE, 'r').read().splitlines()
            filename_queue = tf.train.string_input_producer(filenames)
            reader = tf.WholeFileReader()

            images = []

            for i in range(NUM_DEVICES):
                images.append(
                    tf.train.shuffle_batch(
                        [utils.read_image_op(filename_queue, reader, args.height, args.width)],
                        args.minibatch_size, args.input_queue_max, args.input_queue_min,
                        num_threads=args.input_threads))

        with tf.variable_scope('train'):
            if USE_ADAM:
                d_opt = tf.train.AdamOptimizer(
                    LEARNING_RATE_D, beta1=0.5, beta2=0.9, epsilon=0.001, name='d_opt')
                g_opt = tf.train.AdamOptimizer(
                    LEARNING_RATE_G, beta1=0.5, beta2=0.9, epsilon=0.001, name='g_opt')
            else:
                d_opt = tf.train.RMSPropOptimizer(LEARNING_RATE_D, name='d_opt')
                g_opt = tf.train.RMSPropOptimizer(LEARNING_RATE_G, name='g_opt')

        reuse = False
        tower_outs = {'d': [], 'g': []}
        tower_passed =  {'f': [], 'r': []}
        tower_losses = {'d': [], 'g': []}
        tower_grads = {'d': [], 'g': []}
        feature_size = [args.minibatch_size] + FEATURE_SHAPE

        def calc_passed(x):
            v = tf.cast(x > 0.5, tf.int64)
            x = tf.cast(tf.equal(v, 1), tf.float32)
            return x

        discriminator = model.discriminator
        generator_loss = model.generator_loss
        discriminator_loss = model.discriminator_loss

        for d in range(NUM_DEVICES):
            with tf.device('/gpu:{}'.format(d)), tf.name_scope('model'):
                feature = model.random_feature_generator(feature_size)
                generated_image = model.generator(
                    feature, FIRST_SHAPE, NUM_UPSAMPLE, reuse=reuse,
                    training=True, var_device=VAR_DEVICE)

                out_from_gen = discriminator(
                    generated_image, NUM_DOWNSAMPLE, reuse=reuse,
                    training=True, var_device=VAR_DEVICE)
                out_from_img = discriminator(
                    images[d], NUM_DOWNSAMPLE, reuse=True,
                    training=True, var_device=VAR_DEVICE)
                reuse = True

                tower_passed['f'].append(calc_passed(out_from_gen))
                tower_passed['r'].append(calc_passed(out_from_img))

                tower_outs['g'].append(out_from_gen)
                tower_outs['d'].append(out_from_img)

                # generator loss

                g_loss = generator_loss(
                    tf.ones_like(out_from_gen, tf.float32), out_from_gen)
                tower_losses['g'].append(g_loss)

                g_train_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]
                g_grads = g_opt.compute_gradients(g_loss, var_list=g_train_vars)
                tower_grads['g'].append(g_grads)

                # discriminator loss

                d_loss = discriminator_loss(
                    tf.ones_like(out_from_img, tf.float32), out_from_img)
                g_loss = generator_loss(
                    tf.zeros_like(out_from_gen, tf.float32), out_from_gen)
                d_loss = tf.concat([d_loss,g_loss], axis=0)

                tower_losses['d'].append(d_loss)

                d_train_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]
                d_grads = d_opt.compute_gradients(d_loss, var_list=d_train_vars)
                tower_grads['d'].append(d_grads)

                # summary

                tf.summary.image('generated_image', generated_image,
                    max_outputs=12, collections=['summary_image'])
                    

        with tf.device(VAR_DEVICE):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                g_losses = tf.reduce_mean( tf.concat(tower_losses['g'], axis=0) )
                tf.summary.scalar('g_losses', g_losses)
                d_losses = tf.reduce_mean( tf.concat(tower_losses['d'], axis=0) )
                tf.summary.scalar('d_losses', d_losses)

                f_passed = tf.reduce_mean( tf.concat(tower_passed['f'], axis=0) )
                r_passed = tf.reduce_mean( tf.concat(tower_passed['r'], axis=0) )
                tf.summary.scalar('f_passed', f_passed)
                tf.summary.scalar('r_passed', r_passed)

                tf.summary.scalar('d_train_count', d_train_count)

                g_gradients_averages = average_gradients(tower_grads['g'])
                d_gradients_averages = average_gradients(tower_grads['d'])

                def train_both():
                    with tf.control_dependencies([
                        g_opt.apply_gradients(g_gradients_averages, global_step),
                        d_opt.apply_gradients(d_gradients_averages),
                        tf.assign_add(d_train_count, 1)
                        ]):

                        return tf.no_op()

                def train_g():
                    with tf.control_dependencies([
                        g_opt.apply_gradients(g_gradients_averages, global_step),
                        ]):
                        return tf.no_op()

                train_op = tf.case([
                    (r_passed < D_THRESH, train_both),
                    (f_passed > D_THRESH, train_both),
                    ], default=train_g)

        log_op = tf.summary.merge_all()
        image_log_op = tf.summary.merge(tf.get_collection('summary_image'))

        config_proto = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session( config=config_proto)

        writer = tf.summary.FileWriter('./logs')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()

        training_dir = Path(args.output_model)
        training_dir.mkdir(parents=True, exist_ok=True)

        latest_checkpoint = tf.train.latest_checkpoint(str(args.input_model))

        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)

        writer.add_graph(tf.get_default_graph())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        gs = 0

        for i in range(args.num_iterations):
            gs = tf.train.global_step(sess, global_step)
            if gs % 10000 == 0:
                saver.save(sess, str(training_dir/'model'), global_step=gs)

            if gs % 10 == 0:
                print("global_step = {}".format(gs))
                _, logs, image_logs =  sess.run([train_op, log_op, image_log_op])
                writer.add_summary(logs, gs)
                writer.add_summary(image_logs, gs)
                writer.flush()
            elif False:
                print("global_step = {}".format(gs))
                _, logs =  sess.run([train_op, log_op])
                writer.add_summary(logs, gs)
                writer.flush()
            else:
                _ = sess.run([train_op])

        gs = tf.train.global_step(sess, global_step)
        saver.save(sess, str(training_dir/'model'), global_step=gs)

if __name__ == '__main__':
    main()

