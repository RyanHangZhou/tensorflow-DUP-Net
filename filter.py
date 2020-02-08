import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import scipy.misc
import sys
import pcl
import provider
# import pc_util
import argparse
import h5py
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))


parser = argparse.ArgumentParser(description='PointCloudLibrary example: Remove outliers')
parser.add_argument('--removal', choices=('radius', 'sor'), default='sor', help='Radius Outlier/Statistical Outlier Removal')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--test_path', default='data/lsgan_bro1_nogan2', help='test file path')
parser.add_argument('--filtered_dir', default='data/modelnet40_filtered/filtered_test', help='filter folder path [filter]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
FILTER_DIR = FLAGS.filtered_dir

TEST_FILES = provider.get_file_list(FLAGS.test_path)

def defense_filter():
    file_size = len(TEST_FILES)
    num_batches = file_size

    for fn in range(num_batches):
        current_orig_data, current_data, current_orig_label, current_label = provider.loadAdvDataFile(TEST_FILES[fn])
        current_orig_label = np.squeeze(current_orig_label)
        current_label = np.squeeze(current_label)

        for batch_idx in range(BATCH_SIZE):
            temp_current_data = current_data[batch_idx, :, :]
            temp_current_label = current_label[batch_idx]
            temp_current_orig_data = current_orig_data[batch_idx, :, :]
            temp_current_orig_label = current_orig_label[batch_idx]
            cloud = pcl.PointCloud()
            cloud.from_array(temp_current_data)

            if FLAGS.removal == 'radius':
                outrem = cloud.make_RadiusOutlierRemoval()
                outrem.set_radius_search(0.8)
                outrem.set_MinNeighborsInRadius(2)
                filtered_cloud = outrem.filter()
            elif FLAGS.removal == 'sor':
                outrem = cloud.make_statistical_outlier_filter()
                outrem.set_mean_k(50)
                outrem.set_std_dev_mul_thresh(1.0)
                filtered_cloud = outrem.filter()
            else:
                print("please specify command line arg paramter 'Radius' or 'Condition'")


            filtered_data = np.asarray(filtered_cloud)
            filtered_data = np.expand_dims(filtered_data, axis=0)
            provider.write_h5(FILTER_DIR, temp_current_orig_data, filtered_data, temp_current_orig_label, temp_current_label, fn*BATCH_SIZE+batch_idx)


if __name__=='__main__':
    defense_filter()
    print('Finished.')
