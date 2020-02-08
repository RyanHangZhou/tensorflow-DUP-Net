# #/bin/bash
# CUDA_ROOT=/usr/local/cuda-9.0
# TF_ROOT=/public/zhouhang/anaconda3/lib/python3.6/site-packages/tensorflow
# /usr/local/cuda-9.0/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# #TF 1.8
# g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0



#!/usr/bin/env bash
# export CUDA_PATH=/usr/local/cuda-9.0
# export CXXFLAGS="-std=c++11"
# export CFLAGS="-std=c99"

# export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
# export CPATH=/usr/local/cuda-9.0/include${CPATH:+:${CPATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# nvcc=/usr/local/cuda-9.0/bin/nvcc
# cudainc=/usr/local/cuda-9.0/include/
# cudalib=/usr/local/cuda-9.0/lib64/
# TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# $nvcc tf_sampling_g.cu -c -o tf_sampling_g.cu.o -std=c++11  -I $TF_INC -DGOOGLE_CUDA=1\
# -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# g++ tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -std=c++11 -shared -fPIC -I $TF_INC \
# -I$TF_INC/external/nsync/public -I $cudainc -L$TF_LIB -ltensorflow_framework -lcudart -L $cudalib -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# #/bin/bash
# CUDA_ROOT=/usr/local/cuda-9.0
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# echo $CUDA_ROOT
# echo $TF_INC
# echo $TF_LIB

# $CUDA_ROOT/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# # TF>=1.4.0
# g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -I$CUDA_ROOT/include -lcudart -L$CUDA_ROOT/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#/bin/bash
CUDA_ROOT=/usr/local/cuda-9.0
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_VER=$(python -c 'import tensorflow as tf; print(tf.__version__)')

echo $CUDA_ROOT
echo $TF_INC
echo $TF_LIB
echo $TF_VER

$CUDA_ROOT/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF>=1.4.0
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I$TF_INC/ -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -I$CUDA_ROOT/include -lcudart -L$CUDA_ROOT/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0