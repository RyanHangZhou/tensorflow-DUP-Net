import argparse
import os
import time
import numpy as np
import tensorflow as tf
from glob import glob
from models import punet as MODEL_GEN
from utils import model_utils
from utils import data_provider
from utils import pc_util
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log/punet/model-120', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=2,   help='Upsampling Ratio [default: 2]')
parser.add_argument('--test_path', default='data/modelnet40_filtered', help='test file path')
parser.add_argument('--upsampled_dir', default='data/modelnet40_upsampled/filtered_test', help='upsampled folder path [upsampled]')
parser.add_argument('--dump_dir', default='result', help='dump folder path [dump]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point
UP_RATIO = FLAGS.up_ratio
MODEL_DIR = FLAGS.log_dir
TEST_DIR = FLAGS.test_path
DUMP_DIR = FLAGS.dump_dir
UPSAMPLED_DIR = FLAGS.upsampled_dir

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

def data_duplicate(current_data):
    duplicated_data = current_data
    iter_cp = NUM_POINT//np.shape(current_data)[1]-1
    if(iter_cp>0):
        for i in range(iter_cp):
            duplicated_data = np.concatenate((duplicated_data, current_data), axis=1)

    rem_shape = NUM_POINT-np.shape(duplicated_data)[1]
    if(rem_shape>0):
        duplicated_data = np.concatenate((duplicated_data, current_data[:, 0:rem_shape, :]), axis=1)

    return duplicated_data


def prediction(data_folder=None,show=False,use_normal=False):

    pointclouds_ipt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
    pred, _ = MODEL_GEN.get_gen_model(pointclouds_ipt, is_training=False, scope='generator', bradius=1.0,
                                      reuse=None, use_normal=use_normal, use_bn=False, use_ibn=False, bn_decay=0.95, up_ratio=UP_RATIO)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_DIR)

        file_size = len(TEST_FILES)
        num_batches = file_size

        for fn in range(num_batches):
            print(fn)
            current_orig_data, current_data, current_orig_label, current_label = provider.loadAdvDataFile(TEST_FILES[fn])
            duplicated_data = data_duplicate(current_data)
            pred_pl = sess.run(pred, feed_dict={pointclouds_ipt: duplicated_data})
            provider.write_h5(UPSAMPLED_DIR, current_orig_data, pred_pl, current_orig_label, current_label, fn)


if __name__ == "__main__":
    prediction()
    print("Finished!")
