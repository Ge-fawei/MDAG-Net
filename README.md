# MDAG-Net

This is our Pytorch implementation for MDAG-Net.

# Prerequisites

* Linux or macOS

* Python 3

* CPU or NVIDIA GPU + CUDA CuDNN

# Testing

The datasets used to test are organized [HERE](https://pan.baidu.com/s/1_ekIgH6RjhEXoXiHxUdEHA) (提取码：ltr1). Please uncompress it into the root path. 

The test model in our paper for the CMU-Seasons dataset are found [HERE](https://pan.baidu.com/s/1OaAY-WOgrS3f6qqtlbsIxw) (提取码：fm07). Please make a new directory ./checkpoints under the root path and uncompress it under ./checkpoints. 

## Testing on CMU-Seasons Dataset:

```javascript
python test.py --phase test --name MDAG_CMU --dataroot the/path/to/CMU_urban --n_domains 12 --which_epoch 70 --serial_test --gpu_ids 0 --which_slice XXX --test_using_cos --mean_cos
```

## Testing on RobotCar Dataset:

* Build the feature database:

```javascript
python save_database_feature.py --phase test --name MDAG_Rob --dataroot the/path/to/RobotCar_rear --n_domains 10 --whih_epoch 70 --serial_test --gpu_ids 0 --test_using_cos --mean_cos --which_slice 1
```

* Test the model:
```javascript
python test_robotcar.py --phase test --name MDAG_Rob --dataroot the/path/to/RobotCar_rear --n_domains 10 --which_epoch 70 --serial_test --gpu_ids 0 --which_slice 1 --test_using_cos --mean_cos
```
