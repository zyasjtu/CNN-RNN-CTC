import random
import os
import cv2
import numpy as np

IMG_H = 32
IMG_C = 3


def __write_list(path, lines):
    with open(path, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.writelines(line)


def split_data(ant_path, valid_ratio):
    with open(ant_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        train_list = lines[int(valid_ratio * len(lines)):]
        valid_list = lines[:int(valid_ratio * len(lines))]
        __write_list(ant_path.rstrip(os.path.basename(ant_path)) + 'train_list', train_list)
        __write_list(ant_path.rstrip(os.path.basename(ant_path)) + 'valid_list', valid_list)
        return train_list, valid_list


def __char_to_int(label, char_map):
    int_list = []
    for c in label:
        int_list.append(char_map[c])
    return np.array(int_list)


def __int_to_char(value, char_map):
    for key in char_map.keys():
        if char_map[key] == int(value):
            return str(key)
        elif len(char_map.keys()) == int(value):
            return ""
    raise ValueError('convert {:d} to char failed.'.format(value))


#
def __sparse_convert(sequences, char_map):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        seq = __char_to_int(seq, char_map)
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    dense_shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, dense_shape


def next_batch(img_dir, data_set, batch_size, char_map, start_idx=None):
    if start_idx is None:
        batch = random.sample(data_set, batch_size)
    else:
        batch = data_set[start_idx:start_idx + batch_size]

    max_width = 0
    for line in batch:
        bgr = cv2.imread(os.path.join(img_dir, line.split(' ')[0]))
        h, w, _ = bgr.shape
        height = IMG_H
        width = int(1.0 * w * height / h)
        if max_width < width:
            max_width = width

    img_list = np.zeros(shape=[batch_size, IMG_H, max_width, IMG_C], dtype=np.float32)
    gt_list = []
    len_list = np.ndarray(shape=[batch_size])
    for idx, line in enumerate(batch):
        fn, gt = line.rstrip('\n').split(' ')
        bgr = cv2.imread(os.path.join(img_dir, fn))
        h, w, _ = bgr.shape
        height = IMG_H
        width = int(1.0 * w * height / h)
        img_list[idx, :, :w, :] = cv2.resize(bgr, (width, height)).astype(np.float32)
        gt_list.append(gt)
        len_list[idx] = int(width / 4)

    return img_list, gt_list, __sparse_convert(gt_list, char_map), len_list


#
def dense_convert(sparse_matrix, char_map):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    dense_matrix = len(char_map.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]

    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(__int_to_char(val, char_map))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list


def get_data(ant_path):
    with open(ant_path, mode='r', encoding='utf-8') as f:
        return f.readlines()
