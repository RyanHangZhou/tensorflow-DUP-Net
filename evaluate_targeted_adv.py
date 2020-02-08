import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/modelnet40_pointnet/model.ckpt', help='model checkpoint file path')
parser.add_argument('--num_classes', type=int, default=40, help='number of classes')
parser.add_argument('--test_path', default='/public/zhouhang/3d/3d_adv_methods/data/lsgan_bro1_nogan2', help='test file path')
parser.add_argument('--dump_dir', default='result', help='dump folder path [dump]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_CLASSES = FLAGS.num_classes
MODEL = importlib.import_module(FLAGS.model)
DUMP_DIR = FLAGS.dump_dir
test_dataset_dir = FLAGS.test_path


if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

TEST_FILES = provider.get_file_list(test_dataset_dir)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False
    total_correct = 0
    total_success_rate = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_seen_class_acc = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_acc = [0 for _ in range(NUM_CLASSES)]

    file_size = len(TEST_FILES)
    num_batches = file_size

    for fn in range(num_batches):
        current_orig_data, current_data, current_orig_label, current_label = provider.loadAdvDataFile(TEST_FILES[fn])
        # for batch_idx in range(BATCH_SIZE):
        #     temp_current_orig_data, temp_current_data, temp_current_orig_label, temp_current_label = provider.loadAdvDataFile(TEST_FILES[fn*BATCH_SIZE+batch_idx])
        #     temp_current_data = np.expand_dims(temp_current_data, axis=0)
        #     temp_current_label = np.expand_dims(temp_current_label, axis=0)
        #     if(batch_idx==0):
        #         current_data = temp_current_data
        #         current_label = temp_current_label
        #     else:
        #         current_data = np.concatenate((current_data, temp_current_data), axis=0)
        #         current_label = np.concatenate((current_label, temp_current_label), axis=0)

        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        current_orig_data = current_orig_data[:,0:NUM_POINT,:]
        current_orig_label = np.squeeze(current_orig_label)

        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label)
        total_success_rate += correct
        total_seen += BATCH_SIZE
        for i in range(BATCH_SIZE):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_orig_label,
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_orig_label)
        total_correct += correct
        for i in range(BATCH_SIZE):
            l = current_orig_label[i]
            total_seen_class_acc[l] += 1
            total_correct_class_acc[l] += (pred_val[i] == l)

    log_string('eval attack success rate: %f'% (total_success_rate / float(total_seen)))
    log_string('eval avg class attack success rate: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))
    log_string('eval accuracy: %f'% (correct / float(total_seen)))
    log_string('eval avg class accuracy: %f' % (np.mean(np.array(total_correct_class_acc)/np.array(total_seen_class_acc, dtype=np.float))))


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
