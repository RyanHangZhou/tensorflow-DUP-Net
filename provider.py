import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]
  

def loadDataFile(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    if f.__bool__():
        f.close()
    return (data, label)


def loadDataFile_with_seg(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    if f.__bool__():
        f.close()
    return (data, label, seg)


# def load_h5(h5_filename):
#     f = h5py.File(h5_filename)
#     data = f['data'][:]
#     label = f['label'][...]
#     if f.__bool__():
#         f.close()
#     return (data, label)

def loadAdvDataFile(filename):
    """ Get the data and labels of the h5 file
        Input:
          Directory of the input filename
        Return:
          N1x3 The original data
          N2x3 The adversarial data
          The original label
          The targeted label
    """
    f = h5py.File(filename)
    orig_data = f['orig_data'][:]
    data = f['data'][:]
    orig_label = f['orig_label'][...]
    label = f['label'][...]
    if f.__bool__():
        f.close()
    return (orig_data, data, orig_label, label)


def write_h5(save_dir, data_orig, data, label_orig, label, index):
    """ Write the data and labels into one h5 file
        Input:
          Directory of the output folder
          N1x3 The original data
          N2x3 The adversarial data
          The original label
          The targeted label
    """
    current_dir = save_dir+str(index)+'.h5'
    h5f = h5py.File(current_dir, 'w')
    h5f.create_dataset('orig_data', data=data_orig)
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('orig_label', data=label_orig)
    h5f.create_dataset('label', data=label)
    h5f.close()


def get_file_list(dir):
    """ Get the file directories of the folder
        Input:
          Directory of the input folder
        Return:
          List of the file names in the folder
    """
    total_list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            file_list = os.path.join(root, name)
            total_list.append(file_list)
    return total_list

