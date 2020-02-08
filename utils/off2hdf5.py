import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import pcl
import shutil
import random

# SAMPLING_BIN = '/usr/bin/pcl_mesh_sampling'
SAMPLING_BIN = '/home/chen/Downloads/pcl/build2/bin/pcl_mesh_sampling'

SAMPLING_POINT_NUM = 2048
SAMPLING_LEAF_SIZE = 0.005

# path of meshes to be processed
DATASET_PATH = 'cover'
def export_ply(pc, filename):
	vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	for i in range(pc.shape[0]):
		vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
	ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
	ply_out.write(filename)

# Sample points on the obj shape
def get_sampling_command(obj_filename, ply_filename):
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f' % SAMPLING_LEAF_SIZE
    return cmd

# --------------------------------------------------------------
# Following are the helper functions to load MODELNET40 shapes
# --------------------------------------------------------------

# Read in the list of categories in MODELNET40
def get_category_names():
    shape_names_file = os.path.join(DATASET_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

# Return all the filepaths for the shapes in MODELNET40 
def get_obj_filenames():
    obj_filelist_file = os.path.join(DATASET_PATH, 'filelist.txt')
    obj_filenames = [os.path.join(DATASET_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    print('Got %d obj files in modelnet40.' % len(obj_filenames))
    return obj_filenames

# Helper function to create the father folder and all subdir folders if not exist
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal, 
		data_dtype='float32', label_dtype='uint8', noral_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

# Read numpy array data and label from h5_filename
def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    print(plydata)
    pc = plydata['vertex'].data[:point_num]
    pcb = list(pc)
    for c in pcb:
        pcb[pcb.index(c)] = list(c)

    pc_array = np.array([[x, y, z] for x,y,z,a,b in pc])
    return pc_array

def load_ply_data2(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file)

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF'

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [int(index) for index in face]

            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices

            assert len(face) > 1

            faces.append(face)

        return vertices, faces


# Make up rows for Nxk array
# Input Pad is 'edge' or 'constant'
def pad_arr_rows(arr, row, pad='edge'):
    assert(len(arr.shape) == 2)
    assert(arr.shape[0] <= row)
    assert(pad == 'edge' or pad == 'constant')
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'constant', (0, 0))


if __name__=='__main__':

    MAIN_MODE = 2

    if MAIN_MODE == 0: # convert obj to pcd
        print('Directory of current path: ', BASE_DIR)
        print('Directory of pcl_mesh_sampling: ', SAMPLING_BIN)
        # iteratively load OBJ files and convert to a PCD file
        # cover meshes
        # input_file_dir = '/data/3D_points/data/cover_norm_residual_obj'
        # output_file_dir = '/data/3D_points/data/cover_norm_residual_pcd'
        # stego meshes
        input_file_dir = '/data/3D_points/data/distortion17_residual_obj'
        output_file_dir = '/data/3D_points/data/distortion17_residual_pcd'
        files = os.listdir(input_file_dir)
        print(files)
        print(np.shape(files))
        for file in files:
            if not os.path.isdir(file):
                input_mesh = input_file_dir+'/'+file
                output_mesh = output_file_dir+'/'+os.path.splitext(file)[0]+'.pcd'
                os.system(get_sampling_command(input_mesh, output_mesh))
    elif MAIN_MODE == 1: # split train and test
        # split training and testing set
        # cover data
        input_file_dir = '/data/3D_points/data/cover_norm_residual_pcd'
        input_file_train_dir = '/data/3D_points/data/cover_norm_train'
        input_file_test_dir = '/data/3D_points/data/cover_norm_test'
        files = os.listdir(input_file_dir)
        train_files = random.sample(files, 260)
        test_files = set(files) - set(train_files)
        for file in train_files:
            if not os.path.isdir(file):
                input_mesh = input_file_dir + '/' + file
                output_mesh = input_file_train_dir + '/' + file
                shutil.copyfile(input_mesh, output_mesh)
        for file in test_files:
            if not os.path.isdir(file):
                input_mesh = input_file_dir + '/' + file
                output_mesh = input_file_test_dir + '/' + file
                shutil.copyfile(input_mesh, output_mesh)
        # stego data
        input_file_dir = '/data/3D_points/data/distortion17_residual_pcd'
        input_file_train_dir = '/data/3D_points/data/distortion17_train'
        input_file_test_dir = '/data/3D_points/data/distortion17_test'
        files = os.listdir(input_file_dir)
        train_files = random.sample(files, 260)
        test_files = set(files) - set(train_files)
        for file in train_files:
            if not os.path.isdir(file):
                input_mesh = input_file_dir + '/' + file
                output_mesh = input_file_train_dir + '/' + file
                shutil.copyfile(input_mesh, output_mesh)
        for file in test_files:
            if not os.path.isdir(file):
                input_mesh = input_file_dir + '/' + file
                output_mesh = input_file_test_dir + '/' + file
                shutil.copyfile(input_mesh, output_mesh)
    elif MAIN_MODE == 2: # convert to h5 file
        #covert pcd files to h5 file
        # train data
        h5_filename = 'ply_data_train0.h5'
        comp = []
        input_file_train_dir = '/data/3D_points/data/cover_norm_train'
        pcd_files = os.listdir(input_file_train_dir)
        for file in pcd_files:
            if not os.path.isdir(file):
                input_mesh = input_file_train_dir+'/'+file
                p = np.asarray(pcl.load(input_mesh))
                p0 = np.zeros((SAMPLING_POINT_NUM-len(p), 3))
                p = np.concatenate((p, p0), axis=0)
                comp.append(list(p))
        len_cover = len(comp)
        label0 = list(np.zeros((len(comp), 1)))
        input_file_train_dir = '/data/3D_points/data/distortion17_train'
        pcd_files = os.listdir(input_file_train_dir)
        for file in pcd_files:
            if not os.path.isdir(file):
                input_mesh = input_file_train_dir+'/'+file
                p = np.asarray(pcl.load(input_mesh))
                p0 = np.zeros((SAMPLING_POINT_NUM-len(p), 3))
                p = np.concatenate((p, p0), axis=0)
                comp.append(list(p))

        comp = np.array(comp)
        label1 = list(np.ones((len_cover, 1)))
        label = np.array(label0 + label1)
        save_h5(h5_filename, comp, label, data_dtype='uint8', label_dtype='uint8')

        # test data
        h5_filename = 'ply_data_test0.h5'
        comp = []
        input_file_train_dir = '/data/3D_points/data/cover_norm_test'
        pcd_files = os.listdir(input_file_train_dir)
        for file in pcd_files:
            if not os.path.isdir(file):
                input_mesh = input_file_train_dir+'/'+file
                p = np.asarray(pcl.load(input_mesh))
                p0 = np.zeros((SAMPLING_POINT_NUM-len(p), 3))
                p = np.concatenate((p, p0), axis=0)
                comp.append(list(p))
        len_cover = len(comp)
        label0 = list(np.zeros((len(comp), 1)))
        input_file_train_dir = '/data/3D_points/data/distortion17_test'
        pcd_files = os.listdir(input_file_train_dir)
        for file in pcd_files:
            if not os.path.isdir(file):
                input_mesh = input_file_train_dir+'/'+file
                p = np.asarray(pcl.load(input_mesh))
                p0 = np.zeros((SAMPLING_POINT_NUM-len(p), 3))
                p = np.concatenate((p, p0), axis=0)
                comp.append(list(p))

        comp = np.array(comp)
        label1 = list(np.ones((len_cover, 1)))
        label = np.array(label0 + label1)
        save_h5(h5_filename, comp, label, data_dtype='uint8', label_dtype='uint8')
    else:
        # load h5 file
        s = 1