import sys
sys.path.append("..")
from codes.models.roi_module import RoIPooling2D
from ..utils import array_tool as at
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from codes.EX_CONST import Const
class MbonHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale):
        # n_class includes the background
        super(MbonHead, self).__init__()

        # self.trans_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #                                  nn.ReLU(True),
        #                                  nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #                                  nn.LeakyReLU(True),
        #                                  ).to("cuda:0")
        # self.converter = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1),
        #                                nn.ReLU(True)).cuda()
        self.classifier = nn.Sequential(nn.Linear(25088, 4096, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(4096, 4096, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5, inplace=False),
                                   ).cuda()

        # self.classifier_ang = nn.Sequential(nn.Linear(25088, 1024, bias=True),
        #                            nn.ReLU(inplace=True),
        #                            nn.Dropout(p=0.5, inplace=False),
        #                            nn.Linear(1024, 1024, bias=True),
        #                            nn.ReLU(inplace=True),
        #                            nn.Dropout(p=0.5, inplace=False),
        #                            ).to("cuda:1")

        self.cls_loc = nn.Sequential(nn.Linear(4096, n_class * 4)).cuda()

        self.score = nn.Sequential(nn.Linear(4096, n_class)).cuda()

        # self.ang_regressor = nn.Sequential(nn.Linear(1024, 512),
        #                                    nn.ReLU(True),
        #                                    nn.Dropout(),
        #                                    nn.Linear(512, 512),
        #                                    nn.ReLU(True),
        #                                    nn.Dropout(),
        #                                    nn.Linear(512, 2)).cuda()

        self.orientation = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, Const.bins * 2)  # to get sin and cos
        ).cuda()
        self.confidence = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, Const.bins),
            # nn.Softmax()
            # nn.Sigmoid()
        ).cuda()
        #
        # self.orientation = nn.Sequential(
        #     nn.Linear(25088, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(256, Const.bins * 2)  # to get sin and cos
        # ).cuda()
        # self.confidence = nn.Sequential(
        #     nn.Linear(25088, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(256, Const.bins),
        #     # nn.Softmax()
        #     # nn.Sigmoid()
        # ).cuda()

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        # normal_init(self.ang_regressor, 0, 0.01)
        normal_init(self.orientation, 0, 0.01)
        normal_init(self.confidence, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous().to(x.device)
        # print(x.shape, x.device, indices_and_rois.shape, indices_and_rois.device)
        # x = self.trans_layer(x)
        plt.imsave("imgfeature.jpg", torch.norm(x[0].detach(), dim=0).cpu().numpy())
        pool = self.roi(x, indices_and_rois).cuda()
        # print(pool.shape)
        # pool_multibin = self.converter(pool)
        pool = pool.view(pool.size(0), -1)
        # print(self.classifier)
        # print(pool.shape)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        orientation = self.orientation(pool)
        # orientation = self.orientation(pool_multibin.view(pool_multibin.size(0), -1))
        orientation = orientation.view(-1, Const.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(pool)
        return roi_cls_locs, roi_scores, orientation, confidence


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
