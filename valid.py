import tensorflow as tf
import json
import crnn_model
import numpy as np
import data_processor

batch_size = 32
ckpt_dir = './checkpoint/'


def valid(img_dir, ant_path, map_path):
    valid_list = data_processor.get_data(ant_path)
    char_map = json.load(open(map_path, 'r'))

    img_input = tf.placeholder(tf.float32, shape=[batch_size, data_processor.IMG_H, None, data_processor.IMG_C])
    len_input = tf.placeholder(dtype=tf.int32, shape=[batch_size])

    network = crnn_model.CRNN(training=False,
                              num_classes=len(char_map.keys()) + 1)
    crnn_out = network.build(images=img_input, seq_len=len_input)
    decoded, _ = tf.nn.ctc_beam_search_decoder(crnn_out, len_input, merge_repeated=False)

    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(ckpt_dir)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess=sess, save_path=save_path)

        n_step = len(valid_list) // batch_size
        accuracy = []
        for n in range(n_step):
            img_val, gt_val, gt_sparse_val, len_val = data_processor.next_batch(img_dir, valid_list,
                                                                                batch_size, char_map, n*batch_size)
            preds = sess.run(decoded, feed_dict={img_input: img_val, len_input: len_val})
            preds = data_processor.dense_convert(preds[0], char_map)

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

            for index, img in enumerate(img_val):
                if not (gt_val[index] == preds[index]):
                    print(index+n*batch_size, gt_val[index], preds[index])

        accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        print('Mean valid accuracy is {:5f}'.format(accuracy))


if __name__ == '__main__':
    valid('./data/number_img/', './data/valid_list', './cfg/char_map.json')
