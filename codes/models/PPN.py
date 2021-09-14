import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from codes.models.resnet import resnet18
import matplotlib
from codes.models.region_proposal_network import RegionProposalNetwork
import cv2
from codes.EX_CONST import Const

import matplotlib.pyplot as plt
matplotlib.use('Agg')

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(1, self.shape[0])

class PerspTransDetector(nn.Module):
    def __init__(self, dataset = None):
        super().__init__()
        if dataset is not None:
            self.num_cam = dataset.num_cam
            self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
            imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                               dataset.base.extrinsic_matrices,
                                                                               dataset.base.worldgrid2worldcoord_mat)

            self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
            # img
            self.upsample_shape = list(map(lambda x: int(x / Const.reduce), self.img_shape))
            img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
            img_zoom_mat = np.diag(np.append(img_reduce, [1]))
            # map
            map_zoom_mat = np.diag(np.append(np.ones([2]) / Const.reduce, [1]))
            self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                              for cam in range(self.num_cam)]

        self.backbone = nn.Sequential(*list(resnet18(pretrained=True, replace_stride_with_dilation=[False, False, True]).children())[:-2]).cuda()
        self.rpn = RegionProposalNetwork(in_channels=1026, mid_channels=1026, ratios=[0.9, 1.1], anchor_scales=[4]).cuda()


    def forward(self, imgs,frame, gt_boxes = None, epoch = None, visualize=False, train = True, mark = None):
        B, N, C, H, W = imgs.shape
        world_features = []
        img_featuremap = []

        for cam in range(self.num_cam):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            img_feature =self.backbone(imgs[:, cam].cuda())
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')

            if cam == 0:
                plt.imsave("img_norm_0.jpg", torch.norm(img_feature[0], dim=0).cpu().numpy())
            else:
                plt.imsave("img_norm_1.jpg", torch.norm(img_feature[0], dim=0).cpu().numpy())

            img_featuremap.append(img_feature)

            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().cuda()

            world_feature = kornia.warp_perspective(img_feature.cuda(), proj_mat, self.reducedgrid_shape) # 0.0142 * 2 = 0.028

            world_feature = kornia.vflip(world_feature)
            if cam == 0:
                plt.imsave("world_feature_0.jpg", torch.norm(world_feature[0], dim=0).cpu().numpy())
            else:
                plt.imsave("world_feature_1.jpg", torch.norm(world_feature[0], dim=0).cpu().numpy())
            world_features.append(world_feature.cuda())
        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).cuda()], dim=1)
        plt.imsave("world_features.jpg", torch.norm(world_features[0], dim=0).cpu().numpy())
        rpn_locs, rpn_scores, anchor, rois, roi_indices = self.rpn(world_features, Const.grid_size) # 0.08

        return rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremap, world_features


    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret

def vis_feature(x, max_num=5, out_path='/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/'):
    for i in range(0, x.shape[1]):
        if i >= max_num:
            break
        feature = x[0, i, :, :].view(x.shape[-2], x.shape[-1])
        feature = feature.detach().cpu().numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.round(feature * 255).astype(np.uint8)
        feature_img = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        dst_path = os.path.join(out_path, str(i) + '.jpg')
        cv2.imwrite(dst_path, feature_img)