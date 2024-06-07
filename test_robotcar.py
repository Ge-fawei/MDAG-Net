import os
from options.RobotcarTestOptions import RobotcarTestOptions
from data.data_loader import DataLoad
from model.mynet import Mynet
import torch
import glob
import numpy as np

opt = RobotcarTestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

dataset = DataLoad(opt)
model = Mynet(opt)

split_file = os.path.join(opt.dataroot, 'pose_new_rear.txt')
names = np.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=0, usecols=(0))
with open(split_file, 'r') as f:
    poses = f.read().splitlines()

if opt.mean_cos:
    mean_cos = "meancos"
else:
    mean_cos = "plaincos"

if opt.use_two_stage:
    f = open("result_two_" + opt.name + "_" + opt.name_finer + "_" + opt.dataroot.split('/')[2] + "_env" + str(
        opt.test_condition) + '_' + str(opt.top_n) + "_" + mean_cos + "_" + mean_cos_finer + ".txt", 'w')
else:
    f = open("result_" + opt.name + '_' + str(opt.which_epoch) + '_' + opt.dataroot.split('/')[2] + "_env" + str(
        opt.test_condition) + "_" + mean_cos + ".txt", 'w')

# find query features
robotcar_database = []
nameindex = []
dirs = sorted(glob.glob("./features/s1" + "/*"))
for i, name in enumerate(dirs):
    print(i, name)
    feature_path = np.load(name)
    nameindex.append(name)
    robotcar_database.append(torch.from_numpy(feature_path['arr_0']).view(256, 64, 64))
# test
for i, data in enumerate(dataset):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data, i)

    img_path = model.get_image_paths()  # query image path
    retrieved_path, query_path = model.testrob(nameindex, robotcar_database, img_path, i)

    for k in range(len(names)):
        qpath = names[k].split('/')[-1]
        rpath = retrieved_path.split('/')[-1]
        if qpath.split('.')[0] == rpath.split('.')[0]:
            name_to_be_written = query_path.split('/')[2] + '/' + query_path.split('/')[-1]
            f.write(name_to_be_written + poses[k][len(poses[k].split(' ')[0]):] + '\n')

    print('Now:  %s' % img_path[0].split('/')[-1])

f.close()

