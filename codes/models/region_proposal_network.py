import time
from codes.EX_CONST import Const
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import torch as t
from torch import nn

from .utils.bbox_tools import generate_anchor_base
from .utils.creator_tool import ProposalCreator

class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=1026, mid_channels=2048, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=Const.reduce,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # 生成anchors
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        self.n_anchor = self.anchor_base.shape[0]
        # print(self.anchor_base.shape)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, self.n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, self.n_anchor * 4, 1, 1, 0)

        # torchvision

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        # print(x.shape, img_size)
        # batch_images = torch.zeros((1, 3, img_size[0], img_size[1]))
        # image_sizes = [(img_size[0], img_size[1])]
        # image_list_ = image_list.ImageList(batch_images, image_sizes)
        # test_anchors = self.anchor_generator(image_list_, x)
        # print(test_anchors[0][:20])

        # a = np.zeros((img_size[0] + 100, img_size[1] + 100))
        # img = np.uint8(a)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # for anchor in test_anchors[0][:5]:
        #     x1, y1, x2, y2 = anchor
        #     cv2.rectangle(img, (int(x1+ 50), int(y1+ 50)), (int(x2+ 50), int(y2+ 50)), color=(255, 255, 0))
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/test_anchor.jpg", img)

        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww) # 特征图根据feat stride放大到原图，所以生成的是能覆盖原图的anchor，所以是原始grid大小

        n_anchor = anchor.shape[0] // (hh * ww)
        # print("n_anchor", n_anchor)
        # a = np.zeros((img_size[0], img_size[1]))
        # img = np.uint8(a)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #
        # for anchors in anchor[:]:
        #     y1, x1, y2, x2 = anchors
        #     cv2.rectangle(img, (int(x2), int(y2)),(int(x1), int(y1)), color=(255, 255, 0))
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/anchor.jpg", img)

        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)
        # print("dzcc", rpn_locs.shape)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # print(rpn_locs.shape)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        # s = time.time()
        rois = list()
        roi_indices = list()

        for i in range(n):
            # print("dzcddzzcc", rpn_locs.shape, i,rpn_locs[i].shape )
            roi, roi_origin, order = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor,img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        # e = time.time()
        # print(e - s)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        # 第二个rpn
        #
        # new_rpn_locs = nn.Conv2d(1026, n_anchor * 4, 1, 1, 0).to("cuda:1")(h)
        # # print("dzcccccc", new_rpn_locs.shape)
        # new_rpn_locs = new_rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # new_rpn_scores = nn.Conv2d(1026, n_anchor * 2, 1, 1, 0).to("cuda:1")(h)
        # new_rpn_scores = new_rpn_scores.permute(0, 2, 3, 1).contiguous()
        # new_rpn_softmax_scores = F.softmax(new_rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # new_rpn_fg_scores = new_rpn_softmax_scores[:, :, :, :, 1].contiguous()
        # new_rpn_fg_scores = new_rpn_fg_scores.view(n, -1)
        # new_rpn_scores = new_rpn_scores.view(n, -1, 2)
        #
        # new_rois = list()
        # new_roi_indices = list()
        #
        # for k in range(n):
        #     # print("dzc", new_rpn_locs.shape, roi_origin.shape)
        #     new_roi, _, _ = self.proposal_layer(
        #         new_rpn_locs[k][order].cpu().data.numpy(),
        #         new_rpn_fg_scores[k][order].cpu().data.numpy(),
        #         roi_origin, img_size,
        #         scale=scale
        #     )
        #     new_batch_index = k * np.ones((len(new_roi),), dtype=np.int32)
        #     new_rois.append(new_roi)
        #     new_roi_indices.append(new_batch_index)
        #
        # new_rois = np.concatenate(new_rois, axis=0)
        # new_roi_indices = np.concatenate(new_roi_indices, axis=0)

        # print(len(rpn_locs[roi_indices]), len(new_roi_indices), len(new_rois))
        return rpn_locs, rpn_scores, anchor, rois, roi_indices


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)

    # print(shift_x)

    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if type(m) is not nn.Sequential().__class__:
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()
    else:
        for n in m:
            if type(n) is not nn.Linear(1, 1).__class__:
                continue
            if truncated:
                n.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                n.weight.data.normal_(mean, stddev)
                n.bias.data.zero_()
