import tensorflow as tf
import os
import json
import data_processor
import crnn_model
import time
import numpy as np

ckpt_dir = './checkpoint/'
batch_size = 32


def cal_acc(gt_val, preds):
    accuracy = []
    for index, gt in enumerate(gt_val):
        pred = preds[index]
        total_count = len(gt)
        correct_count = 0
        try:
            for i, tmp in enumerate(gt):
                if tmp == pred[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / total_count)
            except ZeroDivisionError:
                if len(pred) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)
    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    return accuracy


def train(img_dir, ant_path, map_path):
    #
    train_data, valid_data = data_processor.split_data(ant_path, 0.3)
    char_map = json.load(open(map_path, 'r'))
    img_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, data_processor.IMG_H, None, 3])
    gt_input = tf.sparse_placeholder(dtype=tf.int32)
    len_input = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    #
    network = crnn_model.CRNN(training=True,
                              num_classes=len(char_map.keys()) + 1)
    crnn_out = network.build(images=img_input, seq_len=len_input)
    ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=gt_input,
                                             inputs=crnn_out,
                                             sequence_length=len_input,
                                             ignore_longer_outputs_than_inputs=True))
    decoded, _ = tf.nn.ctc_beam_search_decoder(crnn_out, len_input, merge_repeated=False)
    seq_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), gt_input))
    #
    global_step = tf.train.create_global_step()
    learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.8, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss,
                                                                                     global_step=global_step)
    #
    saver = tf.train.Saver()
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model_name = 'crnn_ctc_{:s}.ckpt'.format(str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))))
    model_save_path = os.path.join(ckpt_dir, model_name)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(20000):
            img_batch, gt_batch, gt_sparse_batch, len_batch = data_processor.next_batch(img_dir, train_data,
                                                                                        batch_size, char_map)
            _, cl, lr, sd = sess.run([optimizer, ctc_loss, learning_rate, seq_dist],
                                     feed_dict={img_input: img_batch, gt_input: gt_sparse_batch, len_input: len_batch})
            if (step + 1) % 1000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=step)
            if (step + 1) % 100 == 0:
                img_val, gt_val, gt_sparse_val, len_val = data_processor.next_batch(img_dir, valid_data,
                                                                                    batch_size, char_map)
                preds = sess.run(decoded, feed_dict={img_input: img_val, gt_input: gt_sparse_val, len_input: len_val})
                preds = data_processor.dense_convert(preds[0], char_map)
                accuracy = cal_acc(gt_val, preds)
                print('step:{:d} lr={:9f} cl={:9f} sd={:9f} acc={:9f}'.format(step + 1, lr, cl, sd, accuracy))


if __name__ == '__main__':
    train('./data/number_img', './data/number_list', './cfg/char_map.json')
