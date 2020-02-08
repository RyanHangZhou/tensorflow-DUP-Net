# DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense
Created by [Hang Zhou](http://home.ustc.edu.cn/~zh2991/), [Kejiang Chen](http://home.ustc.edu.cn/~chenkj/), [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html), Han Fang, Wenbo Zhou, [Nenghai Yu](http://staff.ustc.edu.cn/~ynh/).

Introduction
--
This work is published on ICCV, 2019. 

Neural networks are vulnerable to adversarial examples, which poses a threat to their application in security sensitive systems. We propose a Denoiser and UPsampler Network (DUP-Net) structure as defenses for 3D adversarial point cloud classification, where the two modules reconstruct surface smoothness by dropping or adding points. In this paper, statistical outlier removal (SOR) and a data-driven upsampling network are considered as denoiser and upsampler respectively. Compared with baseline defenses, DUP-Net has three advantages. First, with DUP-Net as a defense, the target model is more robust to white-box adversarial attacks. Second, the statistical outlier removal provides added robustness since it is a non-differentiable denoising operation. Third, the upsampler network can be trained on a small dataset and defends well against adversarial attacks generated from other point cloud datasets. We conduct various experiments to validate that DUP-Net is very effective as defense in practice. Our best defense eliminates 83.8% of C&W and l2 loss based attack (point shifting), 50.0% of C&W and Hausdorff distance loss based attack (point adding) and 9.0% of saliency map based attack (point dropping) under 200 dropped points on PointNet.

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

Installation
--
Install TensorFlow. You may also need to install h5py. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7 on Ubuntu 16.04.

Usage
--
To process a point cloud a model by DUP-Net:

    # Statistical outlier removal (SOR)
    python filter.py --removal sor \
                     --batch_size 4 \
                     --test_path 'data/lsgan_bro1_nogan2' \
                     --filtered_dir 'data/modelnet40_filtered/filtered_test'
    
    # DUP-Net
    python upsample.py --num_point 2048 \
                       --up_ratio 2 \
                       --test_path 'data/modelnet40_filtered' \
                       --upsampled_dir 'data/modelnet40_upsampled/upsampled_test'

License
--
