import numpy as np
import torch
import copy, os
import gc
from collections import OrderedDict
from torchvision.transforms import transforms
from util.util import util
from util.image_pool import ImagePool
import glob
import torch.nn.functional as F
import cv2
import random
import math
from skimage import io

from . import mynetwork
from .basemod import Basemod



def norm_image(image):
    """
    :param image: image with [H,W,C]
    :return: image in np uint8
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


class Mynet(Basemod):
    def name(self):
        return 'Mynet'

    def __init__(self, opt):
        super(Mynet, self).__init__(opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None
        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # used metrics
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        self.mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.L2loss = torch.nn.MSELoss()

        # load/define networks
        if self.isTrain:
           self.load_train_epoch = opt.load_train_epoch
           self.train_model = opt.train_model
        which_epoch = opt.which_epoch
        if self.isTrain:
           if self.train_model:
              self.netA = mynetwork.define_A(opt.input_nc, opt.output_nc, opt.ngf,
                                           opt.net_n_blocks, opt.net_n_shared,
                                           self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
              self.netB = mynetwork.define_B(opt.input_nc, opt.output_nc, opt.ngf,
                                       opt.net_n_blocks, opt.net_n_shared,
                                       self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
           else:
              self.netB = mynetwork.define_B(opt.input_nc, opt.output_nc, opt.ngf,
                                       opt.net_n_blocks, opt.net_n_shared,
                                       self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        else:
            self.netA = mynetwork.define_A(opt.input_nc, opt.output_nc, opt.ngf,
                                           opt.net_n_blocks, opt.net_n_shared,
                                           self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
            #self.netB = mynetwork.define_B(opt.input_nc, opt.output_nc, opt.ngf,
                                       #opt.net_n_blocks, opt.net_n_shared,
                                       #self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)

        self.use_two_stage = False
        self.last_retrieval_index_c0 = 0
        self.last_retrieval_index_c1 = 0
        self.last_domain = 0
        self.database = [0] * self.n_domains
        self.netAnewA = None
        self.netAnewB = None
        self.netBnewA = None
        self.netBnewB = None
        self.model = 0
        for i in range(self.n_domains):
            self.database[i] = []


        if self.isTrain:
            blur_fn = lambda x: torch.nn.functional.conv2d(x, self.Tensor(util().gkern_2d()), groups=3, padding=2)
            self.netD = mynetwork.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, blur_fn, opt.norm, self.gpu_ids)


        if self.isTrain: 
           if self.load_train_epoch > 0:
              self.load_network(self.netB, 'B', self.load_train_epoch)
              self.load_network(self.netD, 'D', self.load_train_epoch)
           else:
              if which_epoch > 0:
                 self.load_network(self.netB, 'B', which_epoch)
                 self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            if self.train_model and opt.continue_train:
               self.load_network(self.netA, 'A', which_epoch)
        else:
            self.load_network(self.netA, 'A', which_epoch)
            #self.load_network(self.netB, 'B', which_epoch)

        if not self.isTrain:
            self.test_using_cos = opt.test_using_cos
            # used for retrieval
            self.database_feature_c0 = []
            self.database_path_c0 = []
            self.database_feature_c1 = []
            self.database_path_c1 = []
            self.database_dist_list_c0 = []  # only for visualization
            self.query_feature_list = []
            self.dist_mat_torch = None
            self.robotcar_database = []
            #self.flagA = 0
            #self.flagB = 0

        if self.isTrain:
            self.neg_B = self.Tensor(opt.num_hard_neg, opt.input_nc, opt.fineSize, opt.fineSize)
            self.train_using_cos = opt.train_using_cos
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # used in the adaptive triplet loss
            self.margin = opt.margin
            self.adapt = opt.adapt
            self.margin_sam_triplet = opt.margin_sam_triplet
            self.adapt_sam_triplet = opt.adapt_sam_triplet
            self.use_realAB_as_negative = opt.use_realAB_as_negative
            self.hard_negative = opt.hard_negative
            # define loss functions
            self.criterionCycle = torch.nn.SmoothL1Loss()
            self.criterionGAN = mynetwork.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # initialize optimizers
            if self.isTrain:
               if self.train_model:
                  self.netA.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
                  self.netB.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
                  self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
               else:
                  self.netB.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
                  self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            # Anet
            self.ABloss_triplet = [0] * self.n_domains
            self.loss_tripletA = [0] * self.n_domains
            self.loss_tripletB = [0] * self.n_domains
            self.loss_D = [0] * self.n_domains
            self.loss_GB = [0] * self.n_domains
            self.loss_GA = [0] * self.n_domains
            self.loss_GAB = [0] * self.n_domains
            self.loss_cycleA = [0] * self.n_domains
            self.loss_cycleB = [0] * self.n_domains
            self.newloss_cycleA = [0] * self.n_domains
            self.ABloss_cycle = [0] * self.n_domains
            self.loss_samA = [0] * self.n_domains
            self.loss_samB = [0] * self.n_domains
            self.loss_samfB = [0] * self.n_domains
            self.loss_netAsamfb = [0] * self.n_domains
            self.loss_netBsamfb = [0] * self.n_domains
            self.loss_sam = [0] * self.n_domains
            self.loss_sam_triplet = [0] * self.n_domains
            self.feature_distanceA = [0] * self.n_domains
            self.feature_distanceB = [0] * self.n_domains
            self.feature_cosA = [0] * self.n_domains
            self.feature_cosB = [0] * self.n_domains
            self.lossA = 0
            # Bnet
            self.loss_G = [0] * self.n_domains
            self.loss_cycle = [0] * self.n_domains
            self.newloss_cycle = [0] * self.n_domains
            self.loss_triplet = [0] * self.n_domains
            self.loss_sam = [0] * self.n_domains
            self.loss_sam_triplet = [0] * self.n_domains
            self.feature_distance = [0] * self.n_domains
            self.feature_cos = [0] * self.n_domains
            self.use_cos_latent_with_L2 = opt.use_cos_latent_with_L2
            # initialize loss multipliers
            self.lambda_triplet, self.lambda_cyc, self.lambda_latent = opt.lambda_triplet, opt.lambda_cycle, opt.lambda_latent
            self.lambda_sam, self.lambda_sam_triplet = opt.lambda_sam, opt.lambda_sam_triplet

    def set_input(self, input, i):
        input_A = input['A']
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
            if self.hard_negative:
                self.neg_B = input['neg_B_tensor'][0].cuda()
                self.neg_DB_list = input['neg_DB_list'][0]
        self.image_paths = input['path']
        self.index = i

    def find_query_features(self):
        with torch.no_grad():
            self.labels = ['query_image_%d' % self.DA]
            raw_encoded = self.netA.encode(self.real_A, self.DA)
            encoded = raw_encoded.view(-1)  # encoded_new1

            # image = copy.deepcopy(self.real_A)
            qr_path = copy.deepcopy(self.image_paths[0])
            #if self.use_two_stage:
                #raw_encoded_finer = self.netG_finer.encode(self.real_A, self.DA).view(-1)
            #else:
            raw_encoded_finer = None
            pair = (encoded, qr_path, raw_encoded_finer)
            # if len(list) % 1 == 0:
            self.query_feature_list.append(pair)  # image and coder

    def load_dist_mat(self):
        if self.opt.mean_cos:
            mean_cos = "meancos"
        else:
            mean_cos = "plaincos"
        if self.opt.meancos_finer:
            mean_cos_finer = "meancosfiner"
        else:
            mean_cos_finer = "plaincosfiner"
        if self.opt.only_for_finer:
            self.dirs = sorted(glob.glob("./features_finer/" + self.opt.dataroot.split('/')[3] + "/*"))
        else:
            self.dirs = sorted(glob.glob("./features/s1" + "/*"))
        for i, name in enumerate(self.dirs):
            print(i, name)
            self.robotcar_database.append(torch.from_numpy(np.load(name)['arr_0']).view(256, 64, 64))
        self.dist_mat_torch = torch.from_numpy(
            np.load("dist_mat_cos_" + self.opt.dataroot.split('/')[3] + "_env" + str(
                self.opt.test_condition) + '_' + mean_cos
                    + '_' + mean_cos_finer + ".npz")[
                'arr_0']).cuda()

    def find_dist_mat(self):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        dist_mat = []
        if self.opt.only_for_finer:
            self.dirs = sorted(glob.glob("./features_finer/" + self.opt.dataroot.split('/')[3] + "/*"))
        else:
            self.dirs = sorted(glob.glob("./features/s1" + "/*"))
        for i, name in enumerate(self.dirs):
            print(i, name)
            feature_path = np.load(name)
            self.robotcar_database.append(torch.from_numpy(feature_path['arr_0']).view(256, 64, 64))
            dist_mat_row = []
            for j, query_feat in enumerate(self.query_feature_list):
                if self.opt.mean_cos:
                    dist = 1 - mean_cos(query_feat[0].view(256, -1),
                                        torch.from_numpy(feature_path['arr_0']).cuda().view(256, -1)).mean()
                else:
                    dist = 1 - cos(query_feat[0],
                                   torch.from_numpy(
                                       feature_path['arr_0']).cuda()) * 1  # + L2loss(query_encoded,item[1])*0
                dist_mat_row.append(dist.cpu().numpy().tolist())
            dist_mat.append(dist_mat_row)
        if self.opt.mean_cos:
            mean_cos = "meancos"
        else:
            mean_cos = "plaincos"
        if self.opt.meancos_finer:
            mean_cos_finer = "meancosfiner"
        else:
            mean_cos_finer = "plaincosfiner"
        #np.savez(
            #"dist_mat_cos_" + self.opt.dataroot.split('/')[3] + "_env" + str(self.opt.test_condition) + '_' + mean_cos
            #+ '_' + mean_cos_finer,
            #np.array(dist_mat))
        self.dist_mat_torch = torch.from_numpy(np.array(dist_mat)).cuda()

    def image_retrieval(self, query_encoded, query_path, query_encoded_finer=None, test_index=-1):
        """
        Used to retrieve the target image in the database given the query encoded feature
        :param query_encoded: the query code
        :param query_path: the path of query image
        :param query_encoded_finer: the query code in the finer retrieval model
        :param test_index: the index of input query images when testing
        :return: the retrieved iamge path and the encoded feature in the database
        """
        min_dix = 100000
        path = None
        final_index = 0

        if query_path.split('/')[-1][11] == '0':
            # for c0, camera 0 in the CMU-Seasons dataset
            self.database_dist_list_c0 = []
            for i, db_path in enumerate(self.database_path_c0):
                if self.test_using_cos:
                    # use the cosine retrieval metric
                    if self.opt.mean_cos:
                        dist = -self.mean_cos(query_encoded.view(256, -1),
                                              self.database_feature_c0[i][0].view(256, -1)).mean(0)
                    else:
                        dist = -self.cos(query_encoded.view(-1),
                                         self.database_feature_c0[i][0].view(-1))
                else:
                    # use L2 metric
                    dist = self.L2loss(query_encoded.view(-1), self.database_feature_c0[i][0].view(-1))
                self.database_dist_list_c0.append(dist.item())
                if not self.use_two_stage:
                    if dist < min_dix:
                        min_dix = dist
                        final_index = i
                        path = db_path
                else:
                    # find top N for finer retrieval
                    if dist < top_n_tensor[self.top_n - 1]:
                        top_n_tensor[self.top_n - 1] = dist
                        top_n_index[self.top_n - 1] = i
                        tmp = top_n_tensor.sort()
                        top_n_tensor = tmp[0]
                        top_n_index = top_n_index[tmp[1]]
            if self.use_two_stage:
                # from coarse to fine strategy
                for i in list(range(self.top_n)):
                    if self.test_using_cos:
                        if self.opt.meancos_finer:
                            dist = -self.mean_cos(query_encoded_finer.view(256, -1),
                                                  self.database_feature_c0[top_n_index[i].int()][1].view(256, -1)).mean(0)
                        else:
                            dist = -self.cos(query_encoded_finer.view(-1),
                                             self.database_feature_c0[top_n_index[i].int()][1].view(-1))
                    else:
                        dist = self.L2loss(query_encoded_finer.view(-1),
                                           self.database_feature_c0[top_n_index[i].int()][1].view(-1))
                    if dist < min_dix:
                        min_dix = dist
                        final_index = top_n_index[i].int()
                        path = self.database_path_c0[final_index]
                if self.opt.save_sam_visualization and test_index % 10 == 0:
                    # save the visualized SAM maps
                    self.find_grad_sam(query_encoded_finer, query_path, self.database_feature_c0[
                        self.database_dist_list_c0.index(sorted(self.database_dist_list_c0)[100])][1], test_index, 100)
                    self.find_grad_sam(self.database_feature_c0[
                                           self.database_dist_list_c0.index(sorted(self.database_dist_list_c0)[100])][
                                           1], self.database_path_c0[
                                           self.database_dist_list_c0.index(sorted(self.database_dist_list_c0)[100])],
                                       query_encoded_finer, test_index, 100)
                    self.find_grad_sam(query_encoded_finer, self.image_paths[0],
                                       self.database_feature_c0[final_index][1], test_index)
                    self.find_grad_sam(self.database_feature_c0[final_index][1], path, query_encoded_finer, test_index)
            print("Minimun distance is :", min_dix.item(), " least index: ", final_index)
            print("Retrieved path: ", path.split('/')[-1], " query path: ", query_path.split('/')[-1])
        else:
            for i, db_path in enumerate(self.database_path_c1):
                # for camera 1
                if self.test_using_cos:
                    if self.opt.mean_cos:
                        dist = -self.mean_cos(query_encoded.view(256, -1),
                                         self.database_feature_c1[i][0].view(256, -1)).mean(0)
                    else:
                        dist = -self.cos(query_encoded.view(-1),
                                    self.database_feature_c1[i][0].view(-1))  # + L2loss(query_encoded,item[1])*0
                else:
                    dist = self.L2loss(query_encoded.view(-1), self.database_feature_c1[i][0].view(-1))
                if not self.use_two_stage:
                    if dist < min_dix:
                        min_dix = dist
                        final_index = i
                        path = db_path
                else:
                    if dist < top_n_tensor[self.top_n - 1]:
                        top_n_tensor[self.top_n - 1] = dist
                        top_n_index[self.top_n - 1] = i
                        tmp = top_n_tensor.sort()
                        top_n_tensor = tmp[0]
                        top_n_index = top_n_index[tmp[1]]
            if self.use_two_stage:
                for i in list(range(self.top_n)):
                    if self.test_using_cos:
                        if self.opt.meancos_finer:
                            dist = -self.mean_cos(query_encoded_finer.view(256, -1),
                                             self.database_feature_c1[top_n_index[i].int()][1].view(256, -1)).mean(0)
                        else:
                            dist = -self.cos(query_encoded_finer.view(-1),
                                        self.database_feature_c1[top_n_index[i].int()][1].view(-1))
                    else:
                        dist = self.L2loss(query_encoded_finer.view(-1),
                                      self.database_feature_c1[top_n_index[i].int()][1].view(-1))
                    if dist < min_dix:
                        min_dix = dist
                        final_index = top_n_index[i].int()
                        path = self.database_path_c1[final_index]
            print("Minimun distance is :", min_dix.item(), " least index: ", final_index)
            print("Retrieved path: ", path.split('/')[-1], " query path: ", query_path.split('/')[-1])
        if query_path.split('/')[-1][11] == '0':
            if self.use_two_stage:
                return path, self.database_feature_c0[final_index][1]
            else:
                return path, self.database_feature_c0[final_index][0]
        else:
            if self.use_two_stage:
                return path, self.database_feature_c1[final_index][1]
            else:
                return path, self.database_feature_c1[final_index][0]

    def save_features(self):
        with torch.no_grad():
            self.labels = ['query_image_%d' % self.DA]
            raw_encoded = self.netA.encode(self.real_A, self.DA)

            encoded = raw_encoded.view(-1)  # encoded_new1
            encoded_np = encoded.cpu().numpy()
            db_path = copy.deepcopy(self.image_paths[0])
            if not os.path.exists("./features/" + db_path.split('/')[-3]):
                os.makedirs("./features/" + db_path.split('/')[-3])
            print("./features/" + db_path.split('/')[-3] + '/' + db_path.split('/')[-1][:-4])
            np.savez("./features/" + db_path.split('/')[-3] + '/' + db_path.split('/')[-1][:-4], encoded_np, db_path)
            if self.use_two_stage:
                if not os.path.exists("./features_finer/" + db_path.split('/')[-3]):
                    os.makedirs("./features_finer/" + db_path.split('/')[-3])
                raw_encoded_finer = self.netG_finer.encode(self.real_A, self.DA)
                np.savez("./features_finer/" + db_path.split('/')[-3] + '/' + db_path.split('/')[-1][:-4],
                         raw_encoded_finer.view(-1).cpu().numpy(),
                         db_path)

    def test(self, index=0):
        with torch.no_grad():
            self.visuals = [self.real_A]
            self.labels = ['query_image_%d' % self.DA]
            raw_encoded = self.netA.encode(self.real_A, self.DA)
            #raw_encoded = self.netB.encode(self.real_A, self.DA)
            raw_encoded_finer = None
            if self.DA == 0:
                # building the feature database
                db_path = copy.deepcopy(self.image_paths[0])
                if db_path.split('/')[-1][11] == '0':
                    self.database_feature_c0.append((raw_encoded, raw_encoded_finer))
                    self.database_path_c0.append(db_path)
                else:
                    self.database_feature_c1.append((raw_encoded, raw_encoded_finer))
                    self.database_path_c1.append(db_path)
                return "database"
            else:
                path, retrieved_image = self.image_retrieval(raw_encoded, self.image_paths[0], raw_encoded_finer, index)
                return path

    def testrob(self, nameindex, robotcar_database, img_path, index=0):
        with torch.no_grad():
            self.visuals = [self.real_A]
            self.labels = ['query_image_%d' % self.DA]
            raw_encoded = self.netA.encode(self.real_A, self.DA)
            raw_encoded_finer = None
            if self.DA == 0:
                # building the feature database
                query_path = None
                return "database", query_path
            else:
                path, retrieved_image, query_path = self.image_retrievalrob(raw_encoded, self.image_paths[0], nameindex, robotcar_database, raw_encoded_finer, index)
                return path, query_path

    def image_retrievalrob(self, query_encoded, query_path, nameindex, robotcar_database, query_encoded_finer=None, test_index=-1):
        """
        Used to retrieve the target image in the database given the query encoded feature
        :param query_encoded: the query code
        :param query_path: the path of query image
        :param query_encoded_finer: the query code in the finer retrieval model
        :param test_index: the index of input query images when testing
        :return: the retrieved iamge path and the encoded feature in the database
        """
        min_dix = 100000
        path = None
        final_index = 0

        # for c0, camera 0 in the CMU-Seasons dataset
        self.database_dist_list_c0 = []
        for i, db_path in enumerate(nameindex):
            if self.test_using_cos:
                    # use the cosine retrieval metric
                if self.opt.mean_cos:
                    dist = -self.mean_cos(query_encoded.view(256, -1),
                                              robotcar_database[i].cuda().view(256, -1)).mean(0)
                else:
                    dist = -self.cos(query_encoded.view(-1),
                                         robotcar_database[i].cuda().view(-1))
            else:
                    # use L2 metric
                dist = self.L2loss(query_encoded.view(-1), robotcar_database[i].cuda().view(-1))
            self.database_dist_list_c0.append(dist.item())

            if dist < min_dix:
                min_dix = dist
                final_index = i
                path = db_path
        print("Minimun distance is :", min_dix.item(), " least index: ", final_index)
        print("Retrieved path: ", path.split('/')[-1], " query path: ", query_path.split('/')[-1])
        return path, nameindex[final_index], query_path

    def get_image_paths(self):
        return self.image_paths

    def find_sam_weight(self, query, db):
        mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        mean_cos_similarity = mean_cos(query.view(256, -1), db.view(256, -1)).mean(0)
        grad_map = torch.autograd.grad(mean_cos_similarity, query, create_graph=True)[0]
        weight = grad_map.sum(1).sum(1).view(256, 1, 1).expand([256, 64, 64])
        return weight

    def find_retrieval(self):
        query_path = []
        retrieved_path = []
        if self.use_two_stage:
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
            _, least_dist_index_topN = torch.sort(self.dist_mat_torch, 0)
            least_dist_index_topN = least_dist_index_topN.transpose(0, 1)[:, :self.top_n]
            self.dirs_finer = sorted(glob.glob("./features_finer/" + self.opt.dataroot.split('/')[3] + "/*"))
            for query_index, top_n_index in enumerate(least_dist_index_topN):
                query_feature = self.query_feature_list[query_index][2]
                least_value = 1000000
                path = None
                for _index in top_n_index:
                    if self.opt.mean_cos:
                        dist = 1 - mean_cos(query_feature.view(256, -1),
                                            torch.from_numpy(
                                                np.load(self.dirs_finer[_index.cpu().numpy()])['arr_0']).cuda().view(
                                                256, -1)).mean()
                    else:
                        dist = 1 - cos(query_feature,
                                       torch.from_numpy(
                                           np.load(self.dirs_finer[_index.cpu().numpy()])['arr_0']).cuda()) * 1
                    if dist < least_value:
                        least_value = dist
                        path = self.dirs_finer[_index.cpu().numpy()]
                retrieved_path.append(path)
                query_path.append(self.query_feature_list[query_index][1])
            print("query_path: ", query_path)
            print("retrieved_path: ", retrieved_path)
        else:
            if not self.opt.save_sam_visualization:
                least_dist_index = torch.argmin(self.dist_mat_torch, 0).cpu().numpy()
                for i in list(range(least_dist_index.size)):
                    query_path.append(self.query_feature_list[i][1])
                    retrieved_path.append(self.dirs[least_dist_index[i]])
            else:
                _, least_dist_index_topN = torch.sort(self.dist_mat_torch, 0)
                least_dist_index = least_dist_index_topN.transpose(0, 1)[:, :100].cpu().numpy()
                for i in list(range(least_dist_index.size)):
                    query_path.append(self.query_feature_list[i][1])
                    retrieved_path.append(self.dirs[least_dist_index[i][0]])
                    if i % 10 == 0:
                        self.find_grad_sam(self.query_feature_list[i][0], self.query_feature_list[i][1],
                                           self.robotcar_database[least_dist_index[i][0]], i)
                        self.find_grad_sam(self.robotcar_database[least_dist_index[i][0]],
                                           self.opt.dataroot + "test00/" + self.dirs[least_dist_index[i][0]].split('/')[
                                                                               -1][
                                                                           :-4] + ".jpg", self.query_feature_list[i][0],
                                           i)
                        self.find_grad_sam(self.query_feature_list[i][0], self.query_feature_list[i][1],
                                           self.robotcar_database[least_dist_index[i][99]], i, 100)
                        self.find_grad_sam(self.robotcar_database[least_dist_index[i][99]],
                                           self.opt.dataroot + "test00/" + self.dirs[least_dist_index[i][0]].split('/')[
                                                                               -1][
                                                                           :-4] + ".jpg",
                                           self.query_feature_list[i][99], i,
                                           100)

            print("query_path: ", query_path)
            print("retrieved_path: ", retrieved_path)
        return query_path, retrieved_path


   

    def get_domain(self):
        return self.DA

    

