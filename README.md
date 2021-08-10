# DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense
Created by [Hang Zhou](http://www.sfu.ca/~hza162/), [Kejiang Chen](http://home.ustc.edu.cn/~chenkj/), [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html), [Han Fang](http://home.ustc.edu.cn/~fanghan/), [Wenbo Zhou](http://staff.ustc.edu.cn/~welbeckz/), [Nenghai Yu](http://staff.ustc.edu.cn/~ynh/).

Introduction
--
This repository is for our ICCV 2019 paper [DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_DUP-Net_Denoiser_and_Upsampler_Network_for_3D_Adversarial_Point_Clouds_ICCV_2019_paper.html). 

Installation
--
Install TensorFlow. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7 on Ubuntu 16.04.

Usage
--
Compile sh files in directory "tf_ops/" before usage.

To process a point cloud by DUP-Net:

    # Statistical outlier removal (SOR)
    python filter.py --removal 'sor' \
                     --batch_size 4 \
                     --test_path 'data/lsgan_bro1_nogan2' \
                     --filtered_dir 'data/modelnet40_filtered/filtered_test'
    
    # DUP-Net
    python upsample.py --num_point 2048 \
                       --up_ratio 2 \
                       --test_path 'data/modelnet40_filtered' \
                       --upsampled_dir 'data/modelnet40_upsampled/upsampled_test'

 To classify the processed point cloud:
 
    python evaluate_filtered_targeted_adv.py --num_classes 40 \
                                             --model_path 'log/modelnet40_pointnet/model.ckpt' \
                                             --test_path 'data/modelnet40_upsampled'

Citation
--
If you find our work useful in your research, please consider citing:

    @inproceedings{zhou2019dup,
       title={DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense},
       author={Zhou, Hang and Chen, Kejiang and Zhang, Weiming and Fang, Han and Zhou, Wenbo and Yu, Nenghai},
       booktitle={Proceedings of the IEEE International Conference on Computer Vision},
       pages={1961--1970},
       year={2019}
     }
     
License
--
