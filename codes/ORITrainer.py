import math
import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
# from detectors.evaluation.evaluate import matlab_eval, python_eval
from detectors.evaluation.pyeval.calAOS import evaluateDetectionAPAOS
from detectors.evaluation.evaluate import evaluate
import torch.nn as nn
import warnings
from detectors.loss.gaussian_mse import GaussianMSE
from .models.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator_conf
from .utils import array_tool as at
from codes.EX_CONST import Const
from tensorboardX import SummaryWriter
from detectors.models.utils.bbox_tools import loc2bbox
from PIL import Image
from torchvision.ops import boxes as box_ops
warnings.filterwarnings("ignore")
import time
import cv2

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins


def fix_bn(m):
   classname = m.__class__.__name__
   if classname.find('BatchNorm') != -1:
       m.eval()

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    # print(orient_batch.shape, orientGT_batch.shape, confGT_batch.shape)
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]
    # print("dzc", indexes)
    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    # print("dzc", orient_batch.shape, batch_size, indexes)
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0]) # 每个bin的中心线和实际角度的sin cos
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()

class ORITrainer(BaseTrainer):
    def __init__(self, model, roi_head, denormalize):
        self.model = model
        self.roi_head = roi_head
        self.score_criterion = GaussianMSE().cuda()
        self.denormalize = denormalize
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.SmoothL1Loss()
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator_conf()
        # self.proposal_target_creator_ori = ProposalTargetCreator_ori()
        self.rpn_sigma = 3
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.bins = Const.bins
    def train(self, epoch, data_loader, optimizer, writer):
        self.model.train()
        self.roi_head.train()
        self.model.apply(fix_bn)

        Loss = 0
        RPN_CLS_LOSS = 0
        RPN_LOC_LOSS = 0
        ALL_ROI_CLS_LOSS = 0
        ALL_ANGLE_REG_LOSS = 0
        ALL_ROI_LOC_LOSS = 0
        ALL_ORI_LOSS = 0
        ALL_CONF_LOSS = 0

        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            imgs, bev_xy,bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, left_dirs, right_dirs, left_sincos, right_sincos, left_orientation, right_orientation, left_conf, right_conf, frame, extrin, intrin, mark = data
            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, frame, gt_bbox, mark=mark)
            # print(left_sincos.shape, left_orientation.shape, left_conf.shape)
            # visualize angle
            # bev_img = cv2.imread("/home/dzc/Data/mix/bevimgs/%d.jpg" % frame)
            # for idx, pt in enumerate(bev_xy.squeeze()):
            #     # print("right sin cos", right_sincos)
            #     # print(pt)
            #     x, y = pt[0], pt[1]
            #     cv2.circle(bev_img, (x, y), radius=2, color=(255, 255, 0))
            #     cv2.line(bev_img, (0, Const.grid_height - 1), (x, y), color = (255, 255, 0))
            #     ray = np.arctan(y / (Const.grid_width - x))
            #     theta_l = bev_angle.squeeze()[idx]
            #     theta = theta_l + ray
            #
            #     x1_rot = x - 30
            #     y1_rot = Const.grid_height - y
            #
            #     # print(theta)
            #     nrx = (x1_rot - x) * np.cos(theta) - (y1_rot - (Const.grid_height - y)) * np.sin(theta) + x
            #     nry = (x1_rot - x) * np.sin(theta) + (y1_rot - (Const.grid_height - y)) * np.cos(theta) + (Const.grid_height - y)
            #
            #     # print(x, y, nrx, nry)
            #     cv2.arrowedLine(bev_img, (x, y), (nrx, Const.grid_height - nry), color=(255, 255, 0))
            #     cv2.line(bev_img, (Const.grid_width - 1, 0), (x, y), color = (155, 25, 0))
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/angle.jpg", bev_img)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bbox = gt_bbox[0]
            gt_left_bbox = gt_left_bbox[0]
            gt_right_bbox = gt_right_bbox[0]
            left_dir = left_dirs[0]
            right_dir = right_dirs[0]
            left_orientation = left_orientation[0]
            right_orientation = right_orientation[0]
            left_conf = left_conf[0]
            right_conf = right_conf[0]

            roi = torch.tensor(rois)
            # -----------------RPN Loss----------------------
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(gt_bbox),
                anchor,
                img_size)

            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label,
                self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to(rpn_score.device))

            # ----------------ROI------------------------------
            # 还需要在双视角下的回归gt，以及筛选过后的分类gt，gt_left_loc, gt_left_label, gt_right_loc, gt_right_label
            # print("left_orientation", left_orientation.shape)
            left_2d_bbox, left_sample_roi, left_gt_loc, left_gt_label, left_gt_sincos, left_pos_num, \
            right_2d_bbox,right_sample_roi, right_gt_loc, right_gt_label, right_gt_sincos, right_pos_num, \
            left_gt_orientation, left_gt_conf, right_gt_orientation, right_gt_conf = self.proposal_target_creator(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(left_dir),
                at.tonumpy(right_dir),
                at.tonumpy(left_sincos),
                at.tonumpy(right_sincos),
                at.tonumpy(left_orientation),
                at.tonumpy(right_orientation),
                at.tonumpy(left_conf),
                at.tonumpy(right_conf),
                gt_left_bbox,
                gt_right_bbox,
                extrin, intrin, frame,
                self.loc_normalize_mean,
                self.loc_normalize_std)
            left_sample_roi_index = torch.zeros(len(left_sample_roi))
            right_sample_roi_index = torch.zeros(len(right_sample_roi))
            # print("dzc", left_gt_conf)
            # ---------------------------left_roi_pooling---------------------------------
            left_roi_cls_loc, left_roi_score, left_pred_orientation, left_pred_conf = self.roi_head(
                img_featuremaps[0],
                torch.tensor(left_2d_bbox).to(img_featuremaps[0].device),
                left_sample_roi_index)

            left_n_sample = left_roi_cls_loc.shape[0]
            left_roi_cls_loc = left_roi_cls_loc.view(left_n_sample, -1, 4)
            left_roi_loc = left_roi_cls_loc[torch.arange(0, left_n_sample).long().cuda(), at.totensor(left_gt_label).long()]
            left_gt_label = at.totensor(left_gt_label).long()
            left_gt_loc = at.totensor(left_gt_loc)
            left_gt_orientation = at.totensor(left_gt_orientation)
            left_gt_conf = at.totensor(left_gt_conf)
            left_pred_orientation = left_pred_orientation[:left_pos_num]
            left_pred_conf = left_pred_conf[:left_pos_num]
            # left_sincos_loss = self.MSELoss(left_pred_sincos.float(), torch.tensor(left_gt_sincos).to(left_pred_sincos.device).float())

            # ---------------------------right_roi_pooling---------------------------------
            right_roi_cls_loc, right_roi_score, right_pred_orientation, right_pred_conf = self.roi_head(
                img_featuremaps[1],
                torch.tensor(right_2d_bbox).to(img_featuremaps[1].device),
                right_sample_roi_index)

            right_n_sample = right_roi_cls_loc.shape[0]
            right_roi_cls_loc = right_roi_cls_loc.view(right_n_sample, -1, 4)
            right_roi_loc = right_roi_cls_loc[
                torch.arange(0, right_n_sample).long().cuda(), at.totensor(right_gt_label).long()]
            right_gt_label = at.totensor(right_gt_label).long()
            right_gt_loc = at.totensor(right_gt_loc)
            right_gt_orientation = at.totensor(right_gt_orientation)
            right_gt_conf = at.totensor(right_gt_conf)

            # right_roi_loc_loss = _fast_rcnn_loc_loss(
            #     right_roi_loc.contiguous(),
            #     right_gt_loc,
            #     right_gt_label.data,
            #     1)

            # right_roi_cls_loss = nn.CrossEntropyLoss()(right_roi_score, right_gt_label.to(right_roi_score.device))
            # right_pred_sincos = right_pred_sincos[:right_pos_num]
            right_pred_orientation = right_pred_orientation[:right_pos_num]
            right_pred_conf = right_pred_conf[:right_pos_num]

            all_roi_loc = torch.cat((left_roi_loc, right_roi_loc))
            all_roi_gt_loc = torch.cat((left_gt_loc, right_gt_loc))

            all_roi_score = torch.cat((left_roi_score, right_roi_score))
            all_gt_label = torch.cat((left_gt_label, right_gt_label))

            # all_pred_sincos = torch.cat((left_pred_sincos, right_pred_sincos))
            # all_gt_sincos = torch.cat((torch.tensor(left_gt_sincos), torch.tensor(right_gt_sincos)))

            all_pred_orientation = torch.cat((left_pred_orientation, right_pred_orientation))
            all_gt_orientation = torch.cat((left_gt_orientation, right_gt_orientation))

            all_pred_conf = torch.cat((left_pred_conf, right_pred_conf))
            # print(left_gt_conf)
            all_gt_conf = torch.cat((left_gt_conf, right_gt_conf))

            all_roi_loc_loss = _fast_rcnn_loc_loss(
                all_roi_loc.contiguous(),
                all_roi_gt_loc,
                all_gt_label.data,
                1)

            all_roi_cls_loss = nn.CrossEntropyLoss()(all_roi_score, all_gt_label.to(all_roi_score.device))
            # all_sincos_loss = self.MSELoss(all_pred_sincos.float(), torch.tensor(all_gt_sincos).to(all_pred_sincos.device).float())
            # print(all_pred_orientation.shape, all_gt_orientation.shape, all_gt_conf.shape)
            # all_orientation_loss = 0
            # for i in range(len(all_pred_orientation)):
            all_orientation_loss = OrientationLoss(all_pred_orientation, all_gt_orientation.to(all_pred_orientation.device), all_gt_conf)

            # print(all_gt_conf)10
            all_gt_conf = torch.max(all_gt_conf, dim=1)[1]
            # print("dzc", all_gt_conf)
            all_conf_loss = nn.CrossEntropyLoss()(all_pred_conf, all_gt_conf.to(all_pred_conf.device))

            # print(all_sincos_loss)
            # print(all_pred_sincos, all_gt_sincos)
            # --------------------测试roi pooling------------------------
            # sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator_ori(
            #     roi,
            #     at.tonumpy(gt_bbox),
            #     at.tonumpy(left_dir),
            #     self.loc_normalize_mean,
            #     self.loc_normalize_std)

            # bev_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % frame)
            # for idx, bbxx in enumerate(sample_roi):
            #     # cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0), thickness=1)
            #     cv2.circle(bev_img, (int((bbxx[3] + bbxx[1]) / 2), (int((bbxx[2] + bbxx[0]) / 2))), color=(255, 0, 0), thickness=2, radius=1)
            #     if str(gt_roi_label[idx]) == "0":
            #         cv2.putText(bev_img, str(gt_roi_label[idx]), (int((bbxx[3] + bbxx[1]) / 2), (int((bbxx[2] + bbxx[0]) / 2))),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=(255, 0, 0))
            #     else:
            #         cv2.putText(bev_img, str(gt_roi_label[idx]),
            #                     (int((bbxx[3] + bbxx[1]) / 2), (int((bbxx[2] + bbxx[0]) / 2))),
            #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255))
            # for idx, bbxx in enumerate(gt_bbox):
            #     cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 255), thickness=3)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/roi_img.jpg", bev_img)

            # sample_roi_index = torch.zeros(len(sample_roi))
            # roi_cls_loc, roi_score = self.roi_head(
            #     bev_featuremaps,
            #     sample_roi,
            #     sample_roi_index)
            #
            # n_sample = roi_cls_loc.shape[0]
            # roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            # roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
            #                       at.totensor(gt_roi_label).long()]
            # gt_roi_label = at.totensor(gt_roi_label).long()
            # gt_roi_loc = at.totensor(gt_roi_loc)
            #
            # roi_loc_loss = _fast_rcnn_loc_loss(
            #     roi_loc.contiguous(),
            #     gt_roi_loc,
            #     gt_roi_label.data,
            #     1)

            # roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(roi_score.device))
            # ----------------------Loss-----------------------------
            # loss = rpn_loc_loss * 3 + rpn_cls_loss * 3 + \
            #         (all_roi_loc_loss + all_roi_cls_loss + all_sincos_loss)
            # loss = (rpn_loc_loss + rpn_cls_loss) * 3 + \
            #         (all_roi_loc_loss + all_roi_cls_loss)
            loss = (rpn_loc_loss + rpn_cls_loss) * 0 + \
                    (all_roi_loc_loss * 0 + all_roi_cls_loss * 0 + 0.3 * all_orientation_loss + all_conf_loss)
            Loss += loss.item()


            RPN_CLS_LOSS += rpn_cls_loss.item()
            RPN_LOC_LOSS += rpn_loc_loss.item()
            ALL_ROI_LOC_LOSS += all_roi_loc_loss.item()
            ALL_ROI_CLS_LOSS += all_roi_cls_loss.item()
            ALL_ORI_LOSS += all_orientation_loss.item()
            ALL_CONF_LOSS += all_conf_loss.item()
            # ------------------------------------------------------------
            loss.backward()
            optimizer.step()
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Total Loss", Loss / (batch_idx + 1), niter)
            writer.add_scalar("rpn_loc_loss", RPN_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("rpn_cls_loss", RPN_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("ALL ROI_Loc LOSS", ALL_ROI_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("ALL ROI_Cls LOSS", ALL_ROI_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("ALL ORI LOSS", ALL_ORI_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("ALL_CONF_LOSS", ALL_CONF_LOSS / (batch_idx + 1), niter)

            if batch_idx % 10 == 0:
                print("[Epoch %d] Iter: %d\n" % (epoch, batch_idx),
                      "Total: %4f\n" % (Loss / (batch_idx + 1)),
                      "Rpn Loc : %4f| " % (RPN_LOC_LOSS / (batch_idx + 1)),
                      "Rpn Cls : %4f| " % (RPN_CLS_LOSS / (batch_idx + 1)),
                      "ALL ROI_Loc : %4f| " % ((ALL_ROI_LOC_LOSS) / (batch_idx + 1)),
                      "ALL ROI_Cls : %4f| " % ((ALL_ROI_CLS_LOSS) / (batch_idx + 1)),
                      "ALL ORI : %4f| " % ((ALL_ORI_LOSS) / (batch_idx + 1)),
                      "ALL CONF : %4f| " % ((ALL_CONF_LOSS) / (batch_idx + 1))
                      )
                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    def test(self,epoch, data_loader, writer):
        self.model.train()
        self.model.eval()
        # self.model.eval()
        self.roi_head.eval()

        all_res = []
        all_pred_res = []
        all_gt_res = []

        for batch_idx, data in enumerate(data_loader):
            imgs, bev_xy,bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, left_dirs, right_dirs, left_sincos, right_sincos, left_orientation, right_orientation, left_conf, right_conf, frame, extrin, intrin, mark = data
            total_start = time.time()

            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, frame, mark=mark)
            roi = torch.tensor(rois)

            # -----------投影------------
            # 筛选出来能用的roi，在480、 640内
            # 保留相应的roi和index
            # box转换和保留
            roi_3d = generate_3d_bbox(roi)

            left_2d_bbox = getprojected_3dbox(roi_3d, extrin, intrin, isleft=True)
            right_2d_bbox = getprojected_3dbox(roi_3d, extrin, intrin, isleft=False)

            left_2d_bbox = get_outter(left_2d_bbox)
            right_2d_bbox = get_outter(right_2d_bbox)

            left_index_inside = np.where(
                (left_2d_bbox[:, 0] >= 0) &
                (left_2d_bbox[:, 1] >= 0) &
                (left_2d_bbox[:, 2] <= Const.ori_img_height) &
                (left_2d_bbox[:, 3] <= Const.ori_img_width)
            )[0]

            right_index_inside = np.where(
                (right_2d_bbox[:, 0] >= 0) &
                (right_2d_bbox[:, 1] >= 0) &
                (right_2d_bbox[:, 2] <= Const.ori_img_height) &
                (right_2d_bbox[:, 3] <= Const.ori_img_width)
            )[0]
            if len(right_index_inside) == 0 or len(left_index_inside) == 0:
                continue

            left_2d_bbox = left_2d_bbox[left_index_inside]
            right_2d_bbox = right_2d_bbox[right_index_inside]
            left_rois_indices = roi_indices[left_index_inside]
            right_rois_indices = roi_indices[right_index_inside]

            # left_img = cv2.imread("/home/dzc/Data/mix_simp/img/left1/%d.jpg" % frame)
            # right_img = cv2.imread("/home/dzc/Data/mix_simp/img/right2/%d.jpg" % frame)


            # for car in left_2d_bbox:
            #     # print(car)
            #     xmax = car[3]
            #     xmin = car[1]
            #     ymax = car[2]
            #     ymin = car[0]
            #     cv2.rectangle(left_img, (xmin, ymin), (xmax, ymax), color=(255, 255, 0), thickness=1)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/left_img.jpg", left_img)

            # for car in right_2d_bbox:
            #     xmax = car[3]
            #     xmin = car[1]
            #     ymax = car[2]
            #     ymin = car[0]
            #     cv2.rectangle(right_img, (xmin, ymin), (xmax, ymax), color=(255, 255, 0), thickness=1)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/right_img.jpg", right_img)

            left_2d_bbox = torch.tensor(left_2d_bbox)
            right_2d_bbox = torch.tensor(right_2d_bbox)

            #------------左右ROI pooling-----------
            left_roi_cls_loc, left_roi_score, left_pred_orientation, left_pred_conf = self.roi_head(
                img_featuremaps[0],
                torch.tensor(left_2d_bbox).to(img_featuremaps[0].device),
                left_rois_indices)

            right_roi_cls_loc, right_roi_score, right_pred_orientation, right_pred_conf = self.roi_head(
                img_featuremaps[1],
                torch.tensor(right_2d_bbox).to(img_featuremaps[1].device),
                right_rois_indices)

            # left_roi_cls_loc, left_roi_score, left_pred_sincos = self.roi_head(
            #     img_featuremaps[0],
            #     left_2d_bbox.to(img_featuremaps[0].device),
            #     left_rois_indices)
            # right_roi_cls_loc, right_roi_score, right_pred_sincos = self.roi_head(
            #     img_featuremaps[1],
            #     right_2d_bbox.to(img_featuremaps[1].device),
            #     right_rois_indices)
            # -----------------------NMS---------------------------

            # 需要遍历一下所有的roi
            angle_bins = generate_bins(self.bins)
            tmp = np.zeros(shape=left_pred_conf.shape)
            tmp += angle_bins
            left_angle_bins = tmp

            tmp = np.zeros(shape=right_pred_conf.shape)
            tmp += angle_bins
            right_angle_bins = tmp

            # [R, 2, 2]
            left_argmax = np.argmax(left_pred_conf.detach().cpu().numpy(), axis=1)
            left_orient = left_pred_orientation[np.arange(len(left_pred_orientation)), left_argmax]
            left_cos = left_orient[:, 0]
            left_sin = left_orient[:, 1]
            left_alpha = np.arctan2(left_sin.cpu().detach().numpy(), left_cos.cpu().detach().numpy())
            left_alpha += left_angle_bins[np.arange(len(left_argmax)), left_argmax]   # 0~180, (R, 2), residual angle
            right_argmax = np.argmax(right_pred_conf.detach().cpu().numpy(), axis=1)
            right_orient = right_pred_orientation[np.arange(len(right_pred_orientation)), right_argmax]
            right_cos = right_orient[:, 0]
            right_sin = right_orient[:, 1]
            right_alpha = np.arctan2(right_sin.cpu().detach().numpy(), right_cos.cpu().detach().numpy())
            right_alpha += right_angle_bins[np.arange(len(right_argmax)), right_argmax]   # 0~180, (R, 2), residual angle

            left_prob = at.tonumpy(F.softmax(at.totensor(left_roi_score), dim=1))
            left_front_prob = left_prob[:, 1]
            right_prob = at.tonumpy(F.softmax(at.totensor(right_roi_score), dim=1))
            right_front_prob = right_prob[:, 1]


            position_mark = np.concatenate((np.zeros((left_front_prob.shape[0], )), np.ones((right_front_prob.shape[0]))))
            all_front_prob = np.concatenate((left_front_prob, right_front_prob))
            all_roi_remain = np.concatenate((roi[left_index_inside], roi[right_index_inside]))
            all_pred_alpha = np.concatenate((at.tonumpy(left_alpha), at.tonumpy(right_alpha)))

            # position_mark = np.zeros((left_front_prob.shape[0], ))
            # all_front_prob = left_front_prob
            # all_roi_remain = roi[left_index_inside]
            # all_pred_alpha =at.tonumpy(left_alpha)

            v, indices = torch.tensor(all_front_prob).sort(0)
            indices_remain = indices[v > 0.6]
            print(frame)
            all_roi_remain = all_roi_remain[indices_remain].reshape(len(indices_remain), 4)
            all_pred_alpha = all_pred_alpha[indices_remain].reshape(len(indices_remain), 1)
            all_front_prob = all_front_prob[indices_remain].reshape(len(indices_remain),)
            position_mark = position_mark[indices_remain].reshape(len(indices_remain), 1)
            # test_gt_res, test_pred_res = visualize_3dbox(all_roi_remain, all_pred_alpha, position_mark, gt_bbox,
            #                                              bev_angle, all_front_prob, extrin, intrin, frame, isRoi=True)
            all_bev_boxes = []
            if indices_remain.shape[0] != 0:
                if indices_remain.shape[0] == 1:
                    keep = [0]
                else:
                    keep = box_ops.nms(torch.tensor(all_roi_remain), torch.tensor(all_front_prob), 0)
                all_bev_boxes, all_pred_alpha, position_mark_keep = all_roi_remain[keep].reshape(len(keep), 4), \
                                                                       all_pred_alpha[keep].reshape(len(keep), 1), \
                                                                       position_mark[keep].reshape(len(keep))

            # -----------------------可视化---------------------------
            if len(all_bev_boxes) != 0:
                test_gt_res, test_pred_res = visualize_3dbox(all_bev_boxes, all_pred_alpha, position_mark_keep, gt_bbox, bev_angle, all_front_prob[keep], extrin, intrin, frame)
                # for k, bbox in enumerate(all_bev_boxes):
                #     ymin, xmin, ymax, xmax = bbox
                #     all_res.append([frame, ((xmin + xmax) / 2), ((ymin + ymax) / 2)])

                for p in range(len(test_gt_res)):
                    all_gt_res.append(test_gt_res[p])
                for l in range(len(test_pred_res)):
                    all_pred_res.append(test_pred_res[l])

        res_fpath = '/home/dzc/Data/%s/dzc_res/all_res.txt' % Const.dataset
        all_gt_fpath = '/home/dzc/Data/%s/dzc_res/all_test_gt.txt' % Const.dataset
        # print(all_pred_res)
        all_gt_res = np.array(all_gt_res).reshape(-1, 14)
        all_pred_res = np.array(all_pred_res).reshape(-1, 15)

        np.savetxt(res_fpath, all_pred_res, "%f")
        np.savetxt(all_gt_fpath, all_gt_res, "%f")

        recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(all_gt_fpath),
                                                        data_loader.dataset.base.__name__)

        AP_50, AOS_50, OS_50, AP_25, AOS_25, OS_25 = evaluateDetectionAPAOS(res_fpath, all_gt_fpath)
        print("MODA: ", moda, ", MODP: ", modp, ", Recall: ", recall, ", Prec: ", precision)
        print("AP_50: ", AP_50, " ,AOS_50: ", AOS_50, ", OS_50: ", OS_50)
        print("AP_25: ", AP_25, " ,AOS_25: ", AOS_25, ", OS_25: ", OS_25)

        # print(recall, precision, moda, modp)


    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.roi_head.n_class

def visualize_3dbox(pred_ori, pred_alpha, position_mark, gt_bbox, bev_angle, all_front_prob, extrin, intrin, idx, isRoi=False):
    left_img = cv2.imread("/home/dzc/Data/%s/img/left1/%d.jpg" % (Const.dataset, idx))
    right_img = cv2.imread("/home/dzc/Data/%s/img/right2/%d.jpg" % (Const.dataset, idx))
    mvdet_res = np.loadtxt("/home/dzc/Data/opensource/mvdet_res/res.txt", dtype=np.int)
    mvdet_gt = np.loadtxt('/home/dzc/Data/opensource/mvdet_res/gt.txt', dtype=np.int)
    # right_img = cv2.imread("/home/dzc/Data/mix_simp/img/right2/%d.jpg" % idx)
    # c = 1.5
    # b = 5
    # rows, cols, channels = right_img.shape
    # blank = np.zeros([rows, cols, channels], right_img.dtype)
    # right_img = cv2.addWeighted(right_img, c, blank, 1 - c, b)

    all_pred_res = []
    all_gt_res = []

    n_bbox = pred_ori.shape[0]

    gt_bbox = gt_bbox[0]
    bev_angle = bev_angle[0]
    gt_n_bbox = gt_bbox.shape[0]
    # ---------------------------------------------
    boxes_3d = []
    for j, bbox in enumerate(gt_bbox):
        ymin, xmin, ymax, xmax = bbox
        theta = bev_angle[j]
        # theta = torch.tensor(0)
        center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
        w = 60
        h = 50
        xmin = center_x - w//2
        xmax = center_x + w//2
        ymin = center_y - h//2
        ymax = center_y + h//2

        x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 - 40
        y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height - ymin, Const.grid_height - ymax, Const.grid_height - ymax, Const.grid_height - ymin, (Const.grid_height -ymax + Const.grid_height -ymin) / 2
        

        x1_rot, x2_rot, x3_rot, x4_rot, xmid_rot = \
            int(math.cos(theta) * (x1_ori - center_x) - math.sin(theta) * (
                        y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x2_ori - center_x) - math.sin(theta) * (
                        y2_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x3_ori - center_x) - math.sin(theta) * (
                        y3_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x4_ori - center_x) - math.sin(theta) * (
                        y4_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x_mid - center_x) - math.sin(theta) * (
                        y_mid - (Const.grid_height - center_y)) + center_x)

        y1_rot, y2_rot, y3_rot, y4_rot, ymid_rot = \
            int(math.sin(theta) * (x1_ori - center_x) + math.cos(theta) * (y1_ori - (Const.grid_height - center_y)) + (
                        Const.grid_height - center_y)), \
            int(math.sin(theta) * (x2_ori - center_x) + math.cos(theta) * (y2_ori - (Const.grid_height - center_y)) + (
                        Const.grid_height - center_y)), \
            int(math.sin(theta) * (x3_ori - center_x) + math.cos(theta) * (y3_ori - (Const.grid_height - center_y)) + (
                        Const.grid_height - center_y)), \
            int(math.sin(theta) * (x4_ori - center_x) + math.cos(theta) * (y4_ori - (Const.grid_height - center_y)) + (
                        Const.grid_height - center_y)), \
            int(math.sin(theta) * (x_mid - center_x) + math.cos(theta) * (y_mid - (Const.grid_height - center_y)) + (
                        Const.grid_height - center_y))

        all_gt_res.append([idx.item(), center_x, center_y, w, h, np.rad2deg(theta.item()), x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot])

        pt0 = [x1_rot, y1_rot, 0]
        pt1 = [x2_rot, y2_rot, 0]
        pt2 = [x3_rot, y3_rot, 0]
        pt3 = [x4_rot, y4_rot, 0]
        pt_h_0 = [x1_rot, y1_rot, Const.car_height]
        pt_h_1 = [x2_rot, y2_rot, Const.car_height]
        pt_h_2 = [x3_rot, y3_rot, Const.car_height]
        pt_h_3 = [x4_rot, y4_rot, Const.car_height]
        pt_extra = [xmid_rot, ymid_rot, 0]

        boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3, pt_extra])

    gt_ori = np.array(boxes_3d).reshape((gt_n_bbox, 9, 3))
    gt_projected_2d = getprojected_3dbox(gt_ori, extrin, intrin, isleft=True)
    gt_projected_2d = getprojected_3dbox(gt_ori, extrin, intrin, isleft=False)
    for k in range(gt_n_bbox):
        color = (0, 60, 199)
        cv2.line(right_img, (gt_projected_2d[k][0][0], gt_projected_2d[k][0][1]),
                 (gt_projected_2d[k][1][0], gt_projected_2d[k][1][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][0][0], gt_projected_2d[k][0][1]),
                 (gt_projected_2d[k][3][0], gt_projected_2d[k][3][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][0][0], gt_projected_2d[k][0][1]),
                 (gt_projected_2d[k][4][0], gt_projected_2d[k][4][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][1][0], gt_projected_2d[k][1][1]),
                 (gt_projected_2d[k][5][0], gt_projected_2d[k][5][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][1][0], gt_projected_2d[k][1][1]),
                 (gt_projected_2d[k][2][0], gt_projected_2d[k][2][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][2][0], gt_projected_2d[k][2][1]),
                 (gt_projected_2d[k][3][0], gt_projected_2d[k][3][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][2][0], gt_projected_2d[k][2][1]),
                 (gt_projected_2d[k][6][0], gt_projected_2d[k][6][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][3][0], gt_projected_2d[k][3][1]),
                 (gt_projected_2d[k][7][0], gt_projected_2d[k][7][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][4][0], gt_projected_2d[k][4][1]),
                 (gt_projected_2d[k][5][0], gt_projected_2d[k][5][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][5][0], gt_projected_2d[k][5][1]),
                 (gt_projected_2d[k][6][0], gt_projected_2d[k][6][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][6][0], gt_projected_2d[k][6][1]),
                 (gt_projected_2d[k][7][0], gt_projected_2d[k][7][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][7][0], gt_projected_2d[k][7][1]),
                 (gt_projected_2d[k][4][0], gt_projected_2d[k][4][1]), color=color, thickness=2)
        cv2.line(right_img, (gt_projected_2d[k][7][0], gt_projected_2d[k][7][1]),
                 (gt_projected_2d[k][4][0], gt_projected_2d[k][4][1]), color=color, thickness=2)

        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        #
        cv2.arrowedLine(right_img, (int((gt_projected_2d[k][0][0] + gt_projected_2d[k][2][0]) / 2),
                                    int((gt_projected_2d[k][0][1] + gt_projected_2d[k][2][1]) / 2)),
                        (gt_projected_2d[k][8][0], gt_projected_2d[k][8][1]), color=(255, 60, 199), thickness=2)
    cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d_gt.jpg" % idx, right_img)
    right_img = cv2.imread("/home/dzc/Data/opensource/img/right2/%d.jpg" % (idx))
    boxes_3d = []
    # ============================= MVDET =============================
    cur_res_idx = np.where(mvdet_res[:, 0] == idx.item())[0]
    cur_res = mvdet_res[cur_res_idx][:, 1:]
    # print(mvdet_res[:, 0], idx, mvdet_res[:, 0] == idx)
    for j, bbox in enumerate(cur_res):
        x, y = bbox
        # print(x, y)
        center_x, center_y = x, y
        w, h = 60 / 2, 50 / 2
        xmin = center_x - w
        xmax = center_x + w
        ymin = center_y - h
        ymax = center_y + h

        x1_ori =  (xmin + xmax) / 2 - 25
        y1_ori = Const.grid_height - ymin


        # bev_img = cv2.imread()

        # pt0 = [x1_rot, y1_rot, 0]
        # pt1 = [x2_rot, y2_rot, 0]
        # pt2 = [x3_rot, y3_rot, 0]
        # pt3 = [x4_rot, y4_rot, 0]
        # pt_h_0 = [x1_rot, y1_rot, Const.car_height]
        # pt_h_1 = [x2_rot, y2_rot, Const.car_height]
        # pt_h_2 = [x3_rot, y3_rot, Const.car_height]
        # pt_h_3 = [x4_rot, y4_rot, Const.car_height]
        # pt_extra = [xmid_rot, ymid_rot, 0]


        x1_rot, x2_rot, x3_rot, x4_rot, x5_rot, x6_rot, x7_rot, x8_rot, x9_rot, x10_rot, x11_rot = \
            int(math.cos(np.pi / 6) * (x1_ori - center_x) - math.sin(np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(2*np.pi / 6) * (x1_ori - center_x) - math.sin(2*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(3*np.pi / 6) * (x1_ori - center_x) - math.sin(3*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(4*np.pi / 6) * (x1_ori - center_x) - math.sin(4*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(5*np.pi / 6) * (x1_ori - center_x) - math.sin(5*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(6*np.pi / 6) * (x1_ori - center_x) - math.sin(6*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(7*np.pi / 6) * (x1_ori - center_x) - math.sin(7*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(8*np.pi / 6) * (x1_ori - center_x) - math.sin(8*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(9*np.pi / 6) * (x1_ori - center_x) - math.sin(9*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(10*np.pi / 6) * (x1_ori - center_x) - math.sin(10*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(11*np.pi / 6) * (x1_ori - center_x) - math.sin(11*np.pi / 6) * (
                    y1_ori - (Const.grid_height - center_y)) + center_x)
        #
        y1_rot, y2_rot, y3_rot, y4_rot,  y5_rot, y6_rot, y7_rot, y8_rot, y9_rot, y10_rot,y11_rot = \
            int(math.sin(np.pi / 6) * (x1_ori - center_x) + math.cos(np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(2*np.pi / 6) * (x1_ori - center_x) + math.cos(2*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(3*np.pi / 6) * (x1_ori - center_x) + math.cos(3*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(4*np.pi / 6) * (x1_ori - center_x) + math.cos(4*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(5*np.pi / 6) * (x1_ori - center_x) + math.cos(5*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(6*np.pi / 6) * (x1_ori - center_x) + math.cos(6*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(7*np.pi / 6) * (x1_ori - center_x) + math.cos(7*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(8*np.pi / 6) * (x1_ori - center_x) + math.cos(8*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(9*np.pi / 6) * (x1_ori - center_x) + math.cos(9*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(10*np.pi / 6) * (x1_ori - center_x) + math.cos(10*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y)), \
            int(math.sin(11*np.pi / 6) * (x1_ori - center_x) + math.cos(11*np.pi / 6) * (y1_ori - (Const.grid_height - center_y)) + (
                    Const.grid_height - center_y))

        cy0 = [x1_ori, y1_ori, 0]
        cy1 = [x1_ori, y1_ori, Const.car_height]
        cy2 = [x1_rot, y1_rot, 0]
        cy3 = [x1_rot, y1_rot, Const.car_height]
        cy4 = [x2_rot, y2_rot, 0]
        cy5 = [x2_rot, y2_rot, Const.car_height]
        cy6 = [x3_rot, y3_rot, 0]
        cy7 = [x3_rot, y3_rot, Const.car_height]
        cy8 = [x4_rot, y4_rot, 0]
        cy9 = [x4_rot, y4_rot, Const.car_height]
        cy10 = [x5_rot, y5_rot, 0]
        cy11 = [x5_rot, y5_rot, Const.car_height]
        cy12 = [x6_rot, y6_rot, 0]
        cy13 = [x6_rot, y6_rot, Const.car_height]
        cy14 = [x7_rot, y7_rot, 0]
        cy15 = [x7_rot, y7_rot, Const.car_height]
        cy16 = [x8_rot, y8_rot, 0]
        cy17 = [x8_rot, y8_rot, Const.car_height]
        cy18 = [x9_rot, y9_rot, 0]
        cy19 = [x9_rot, y9_rot, Const.car_height]
        cy20 = [x10_rot, y10_rot, 0]
        cy21 = [x10_rot, y10_rot, Const.car_height]
        cy22 = [x11_rot, y11_rot, 0]
        cy23 = [x11_rot, y11_rot, Const.car_height]


        boxes_3d.append([cy0, cy1, cy2, cy3, cy4, cy5, cy6, cy7, cy8, cy9, cy10, cy11, cy12,cy13, cy14,cy15,cy16,cy17,cy18,cy19,cy20,cy21, cy22, cy23])
    # print(j)
    gt_ori = np.array(boxes_3d).reshape((j+1, 24, 3))
    gt_projected_2d = getprojected_3dbox(gt_ori, extrin, intrin, isleft=True)
    gt_projected_2d = getprojected_3dbox(gt_ori, extrin, intrin, isleft=False)
    for k in range(j+1):
        color = (0, 255, 255)
        cv2.line(right_img, (gt_projected_2d[k][0][0], gt_projected_2d[k][0][1]),
                 (gt_projected_2d[k][2][0], gt_projected_2d[k][2][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][2][0], gt_projected_2d[k][2][1]),
                 (gt_projected_2d[k][4][0], gt_projected_2d[k][4][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][4][0], gt_projected_2d[k][4][1]),
                 (gt_projected_2d[k][6][0], gt_projected_2d[k][6][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][6][0], gt_projected_2d[k][6][1]),
                 (gt_projected_2d[k][8][0], gt_projected_2d[k][8][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][8][0], gt_projected_2d[k][8][1]),
                 (gt_projected_2d[k][10][0], gt_projected_2d[k][10][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][10][0], gt_projected_2d[k][10][1]),
                 (gt_projected_2d[k][12][0], gt_projected_2d[k][12][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][12][0], gt_projected_2d[k][12][1]),
                 (gt_projected_2d[k][14][0], gt_projected_2d[k][14][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][14][0], gt_projected_2d[k][14][1]),
                 (gt_projected_2d[k][16][0], gt_projected_2d[k][16][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][16][0], gt_projected_2d[k][16][1]),
                 (gt_projected_2d[k][18][0], gt_projected_2d[k][18][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][18][0], gt_projected_2d[k][18][1]),
                 (gt_projected_2d[k][20][0], gt_projected_2d[k][20][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][20][0], gt_projected_2d[k][20][1]),
                 (gt_projected_2d[k][22][0], gt_projected_2d[k][22][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][22][0], gt_projected_2d[k][22][1]),
                 (gt_projected_2d[k][0][0], gt_projected_2d[k][0][1]), color=color, thickness=1)

        cv2.line(right_img, (gt_projected_2d[k][0+1][0], gt_projected_2d[k][0+1][1]),
                 (gt_projected_2d[k][2+1][0], gt_projected_2d[k][2+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][2+1][0], gt_projected_2d[k][2+1][1]),
                 (gt_projected_2d[k][4+1][0], gt_projected_2d[k][4+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][4+1][0], gt_projected_2d[k][4+1][1]),
                 (gt_projected_2d[k][6+1][0], gt_projected_2d[k][6+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][6+1][0], gt_projected_2d[k][6+1][1]),
                 (gt_projected_2d[k][8+1][0], gt_projected_2d[k][8+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][8+1][0], gt_projected_2d[k][8+1][1]),
                 (gt_projected_2d[k][10+1][0], gt_projected_2d[k][10+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][10][0], gt_projected_2d[k][10+1][1]),
                 (gt_projected_2d[k][12+1][0], gt_projected_2d[k][12+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][12+1][0], gt_projected_2d[k][12+1][1]),
                 (gt_projected_2d[k][14+1][0], gt_projected_2d[k][14+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][14+1][0], gt_projected_2d[k][14+1][1]),
                 (gt_projected_2d[k][16+1][0], gt_projected_2d[k][16+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][16+1][0], gt_projected_2d[k][16+1][1]),
                 (gt_projected_2d[k][18+1][0], gt_projected_2d[k][18+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][18+1][0], gt_projected_2d[k][18+1][1]),
                 (gt_projected_2d[k][20+1][0], gt_projected_2d[k][20+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][20+1][0], gt_projected_2d[k][20+1][1]),
                 (gt_projected_2d[k][22+1][0], gt_projected_2d[k][22+1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][22+1][0], gt_projected_2d[k][22+1][1]),
                 (gt_projected_2d[k][0+1][0], gt_projected_2d[k][0+1][1]), color=color, thickness=1)

        cv2.line(right_img, (gt_projected_2d[k][0][0], gt_projected_2d[k][0][1]),
                 (gt_projected_2d[k][1][0], gt_projected_2d[k][1][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][6][0], gt_projected_2d[k][6][1]),
                 (gt_projected_2d[k][7][0], gt_projected_2d[k][7][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][12][0], gt_projected_2d[k][12][1]),
                 (gt_projected_2d[k][13][0], gt_projected_2d[k][13][1]), color=color, thickness=1)
        cv2.line(right_img, (gt_projected_2d[k][18][0], gt_projected_2d[k][18][1]),
                 (gt_projected_2d[k][19][0], gt_projected_2d[k][19][1]), color=color, thickness=1)
        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        #
        # cv2.arrowedLine(right_img, (int((gt_projected_2d[k][0][0] + gt_projected_2d[k][2][0]) / 2),
        #                             int((gt_projected_2d[k][0][1] + gt_projected_2d[k][2][1]) / 2)),
        #                 (gt_projected_2d[k][8][0], gt_projected_2d[k][8][1]), color=(255, 60, 199), thickness=2)
        # cv2.line(left_img, (int((projected_2d[k][0+ 9][0] + projected_2d[k][2+ 9][0]) / 2), int((projected_2d[k][0+ 9][1] + projected_2d[k][2+ 9][1]) / 2)), (projected_2d[k][8+ 9][0], projected_2d[k][8+ 9][1]), color = (255, 60, 199), thickness=2)
    cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d_mvdet.jpg" % idx, right_img)
    right_img = cv2.imread("/home/dzc/Data/%s/img/right2/%d.jpg" % (Const.dataset, idx))


    boxes_3d = []
    # print(all_front_prob.shape, pred_angle.shape, pred_ori.shape)
    for i, bbox in enumerate(pred_ori):
        ymin, xmin, ymax, xmax = bbox
        alpha = pred_alpha[i]
        if pred_alpha.shape[0] == 1:
            score = 1.0
        else:
            score = all_front_prob[i]
        if position_mark[i] == 0:
            center_x, center_y = int((bbox[1] + bbox[3]) // 2), int((bbox[0] + bbox[2]) // 2)
            w, h = 60/2, 50/2
            xmin = center_x - w
            xmax = center_x + w
            ymin = center_y - h
            ymax = center_y + h
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 + 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height -ymin, Const.grid_height -ymax, Const.grid_height -ymax, Const.grid_height -ymin, (Const.grid_height -ymax + Const.grid_height -ymin) / 2
            center_x, center_y = int((bbox[1] + bbox[3]) // 2), int((bbox[0] + bbox[2]) // 2)
            ray = np.arctan((Const.grid_height - center_y) / center_x)
            angle = alpha
            # if np.sin(alpha) > 0 and \
            #         np.cos(alpha) < 0:
            #     angle += np.pi
            # elif np.sin(alpha) < 0 and \
            #         np.cos(alpha) < 0:
            #     angle += np.pi
            # elif np.sin(alpha) < 0 and \
            #         np.cos(alpha) > 0:
            #     angle += 2 * np.pi
        else:
            center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
            w, h = 60/2, 50/2
            xmin = center_x - w
            xmax = center_x + w
            ymin = center_y - h
            ymax = center_y + h
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 - 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height -ymin, Const.grid_height -ymax, Const.grid_height -ymax, Const.grid_height -ymin, (Const.grid_height -ymax + Const.grid_height -ymin) / 2
            center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
            ray = np.arctan(center_y / (Const.grid_width - center_x))
            angle = alpha
            # if np.sin(alpha) > 0 and \
            #         np.cos(alpha) < 0:
            #     angle += np.pi
            # elif np.sin(alpha) < 0 and \
            #         np.cos(alpha) < 0:
            #     angle += np.pi
            # elif np.sin(alpha) < 0 and \
            #         np.cos(alpha) > 0:
            #     angle += 2 * np.pi
            # angle += np.pi
        theta_l = angle
        theta = theta_l + ray
        # theta = torch.tensor(0)

        x1_rot, x2_rot, x3_rot, x4_rot, xmid_rot = \
            int(math.cos(theta) * (x1_ori - center_x) - math.sin(theta) * (y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x2_ori - center_x) - math.sin(theta) * (y2_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x3_ori - center_x) - math.sin(theta) * (y3_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x4_ori - center_x) - math.sin(theta) * (y4_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x_mid - center_x) - math.sin(theta) * (y_mid - (Const.grid_height - center_y)) + center_x)

        y1_rot, y2_rot, y3_rot, y4_rot, ymid_rot = \
            int(math.sin(theta) * (x1_ori - center_x) + math.cos(theta) * (y1_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x2_ori - center_x) + math.cos(theta) * (y2_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x3_ori - center_x) + math.cos(theta) * (y3_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x4_ori - center_x) + math.cos(theta) * (y4_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x_mid - center_x) + math.cos(theta) * (y_mid - (Const.grid_height - center_y)) + (Const.grid_height - center_y))

        if position_mark[i] == 0:
            theta_left = theta + np.pi
            all_pred_res.append([idx.item(), center_x, center_y, w, h,  np.rad2deg(theta_left.item()), score, x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot])
        else:
            all_pred_res.append([idx.item(), center_x, center_y, w, h, np.rad2deg(theta.item()), score, x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot])

        pt0 = [x1_rot, y1_rot, 0]
        pt1 = [x2_rot, y2_rot, 0]
        pt2 = [x3_rot, y3_rot, 0]
        pt3 = [x4_rot, y4_rot, 0]
        pt_h_0 = [x1_rot, y1_rot, Const.car_height]
        pt_h_1 = [x2_rot, y2_rot, Const.car_height]
        pt_h_2 = [x3_rot, y3_rot, Const.car_height]
        pt_h_3 = [x4_rot, y4_rot, Const.car_height]
        pt_extra = [xmid_rot, ymid_rot, 0]

        boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3, pt_extra])
    pred_ori = np.array(boxes_3d).reshape((n_bbox, 9, 3))
    projected_2d_left = getprojected_3dbox(pred_ori, extrin, intrin, isleft=True)
    projected_2d = getprojected_3dbox(pred_ori, extrin, intrin, isleft=False)
    for k in range(n_bbox):
        if position_mark[k] == 0:
            color = (255, 255, 0)
            color = (0, 255, 0)
        else:
            color = (100, 100, 200)
            color = (0, 255, 0)
        if position_mark[k] == 1 and isRoi:
            cv2.line(right_img, (projected_2d[k][0][0], projected_2d[k][0][1]), (projected_2d[k][1][0], projected_2d[k][1][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][0][0], projected_2d[k][0][1]), (projected_2d[k][3][0], projected_2d[k][3][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][0][0], projected_2d[k][0][1]), (projected_2d[k][4][0], projected_2d[k][4][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][1][0], projected_2d[k][1][1]), (projected_2d[k][5][0], projected_2d[k][5][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][1][0], projected_2d[k][1][1]), (projected_2d[k][2][0], projected_2d[k][2][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][2][0], projected_2d[k][2][1]), (projected_2d[k][3][0], projected_2d[k][3][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][2][0], projected_2d[k][2][1]), (projected_2d[k][6][0], projected_2d[k][6][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][3][0], projected_2d[k][3][1]), (projected_2d[k][7][0], projected_2d[k][7][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][4][0], projected_2d[k][4][1]), (projected_2d[k][5][0], projected_2d[k][5][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][5][0], projected_2d[k][5][1]), (projected_2d[k][6][0], projected_2d[k][6][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][6][0], projected_2d[k][6][1]), (projected_2d[k][7][0], projected_2d[k][7][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][7][0], projected_2d[k][7][1]), (projected_2d[k][4][0], projected_2d[k][4][1]), color = color, thickness=2)
            cv2.line(right_img, (projected_2d[k][7][0], projected_2d[k][7][1]), (projected_2d[k][4][0], projected_2d[k][4][1]), color = color, thickness=2)

            cv2.arrowedLine(right_img, (int((projected_2d[k][0][0] + projected_2d[k][2][0]) / 2),
                                        int((projected_2d[k][0][1] + projected_2d[k][2][1]) / 2)),
                            (projected_2d[k][8][0], projected_2d[k][8][1]), color=(255, 60, 199), thickness=2)
        elif position_mark[k] == 0 and isRoi:
            cv2.line(left_img, (projected_2d_left[k][0][0], projected_2d_left[k][0][1]), (projected_2d_left[k][1][0], projected_2d_left[k][1][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][0][0], projected_2d_left[k][0][1]), (projected_2d_left[k][3][0], projected_2d_left[k][3][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][0][0], projected_2d_left[k][0][1]), (projected_2d_left[k][4][0], projected_2d_left[k][4][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][1][0], projected_2d_left[k][1][1]), (projected_2d_left[k][5][0], projected_2d_left[k][5][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][1][0], projected_2d_left[k][1][1]), (projected_2d_left[k][2][0], projected_2d_left[k][2][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][2][0], projected_2d_left[k][2][1]), (projected_2d_left[k][3][0], projected_2d_left[k][3][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][2][0], projected_2d_left[k][2][1]), (projected_2d_left[k][6][0], projected_2d_left[k][6][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][3][0], projected_2d_left[k][3][1]), (projected_2d_left[k][7][0], projected_2d_left[k][7][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][4][0], projected_2d_left[k][4][1]), (projected_2d_left[k][5][0], projected_2d_left[k][5][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][5][0], projected_2d_left[k][5][1]), (projected_2d_left[k][6][0], projected_2d_left[k][6][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][6][0], projected_2d_left[k][6][1]), (projected_2d_left[k][7][0], projected_2d_left[k][7][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][7][0], projected_2d_left[k][7][1]), (projected_2d_left[k][4][0], projected_2d_left[k][4][1]), color = color, thickness=2)
            cv2.line(left_img, (projected_2d_left[k][7][0], projected_2d_left[k][7][1]), (projected_2d_left[k][4][0], projected_2d_left[k][4][1]), color = color, thickness=2)

            cv2.arrowedLine(left_img, (int((projected_2d_left[k][0][0] + projected_2d_left[k][2][0]) / 2), int((projected_2d_left[k][0][1] + projected_2d_left[k][2][1]) / 2)), (projected_2d_left[k][8][0], projected_2d_left[k][8][1]), color = (255, 60, 199), thickness=2)
        # cv2.line(left_img, (int((projected_2d[k][0+ 9][0] + projected_2d[k][2+ 9][0]) / 2), int((projected_2d[k][0+ 9][1] + projected_2d[k][2+ 9][1]) / 2)), (projected_2d[k][8+ 9][0], projected_2d[k][8+ 9][1]), color = (255, 60, 199), thickness=2)
        else:
            cv2.line(right_img, (projected_2d[k][0][0], projected_2d[k][0][1]),
                     (projected_2d[k][1][0], projected_2d[k][1][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][0][0], projected_2d[k][0][1]),
                     (projected_2d[k][3][0], projected_2d[k][3][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][0][0], projected_2d[k][0][1]),
                     (projected_2d[k][4][0], projected_2d[k][4][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][1][0], projected_2d[k][1][1]),
                     (projected_2d[k][5][0], projected_2d[k][5][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][1][0], projected_2d[k][1][1]),
                     (projected_2d[k][2][0], projected_2d[k][2][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][2][0], projected_2d[k][2][1]),
                     (projected_2d[k][3][0], projected_2d[k][3][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][2][0], projected_2d[k][2][1]),
                     (projected_2d[k][6][0], projected_2d[k][6][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][3][0], projected_2d[k][3][1]),
                     (projected_2d[k][7][0], projected_2d[k][7][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][4][0], projected_2d[k][4][1]),
                     (projected_2d[k][5][0], projected_2d[k][5][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][5][0], projected_2d[k][5][1]),
                     (projected_2d[k][6][0], projected_2d[k][6][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][6][0], projected_2d[k][6][1]),
                     (projected_2d[k][7][0], projected_2d[k][7][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][7][0], projected_2d[k][7][1]),
                     (projected_2d[k][4][0], projected_2d[k][4][1]), color=color, thickness=2)
            cv2.line(right_img, (projected_2d[k][7][0], projected_2d[k][7][1]),
                     (projected_2d[k][4][0], projected_2d[k][4][1]), color=color, thickness=2)

            cv2.arrowedLine(right_img, (int((projected_2d[k][0][0] + projected_2d[k][2][0]) / 2),
                                        int((projected_2d[k][0][1] + projected_2d[k][2][1]) / 2)),
                            (projected_2d[k][8][0], projected_2d[k][8][1]), color=(255, 60, 199), thickness=2)

    if isRoi:
        cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d_roi.jpg" % idx, right_img)
        cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d_left_roi.jpg" % idx, left_img)
    else:
        cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d.jpg" % idx, right_img)
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d_left.jpg" % idx, left_img)
    return all_gt_res, all_pred_res

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    # print(type(x), type(t), type(in_weight))
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).to(pred_loc.device)
    gt_loc = torch.tensor(gt_loc).to(pred_loc.device)
    gt_label = torch.tensor(gt_label).to(pred_loc.device)

    # print(in_weight.shape, gt_loc.shape, gt_label.shape)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    # print(gt_label)
    in_weight[(torch.tensor(gt_label) > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # loc_loss = F.smooth_l1_loss(pred_loc, gt_loc)

    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss

def generate_3d_bbox(pred_bboxs):
    # 输出以左下角为原点的3d坐标
    n_bbox = pred_bboxs.shape[0]

    zeros = np.zeros((n_bbox, 1))
    heights = np.zeros((n_bbox, 1)) * Const.car_height
    ymax, xmax, ymin, xmin = pred_bboxs[:, 0].reshape(-1, 1), pred_bboxs[:, 1].reshape(-1, 1), pred_bboxs[:, 2].reshape(-1, 1), pred_bboxs[:, 3].reshape(-1, 1)

    pt0s = np.concatenate((xmax, Const.grid_height - ymin, zeros), axis=1).reshape(1, n_bbox, 3)
    pt1s = np.concatenate((xmin, Const.grid_height - ymin, zeros), axis=1).reshape(1, n_bbox, 3)
    pt2s = np.concatenate((xmin, Const.grid_height - ymax, zeros), axis=1).reshape(1, n_bbox, 3)
    pt3s = np.concatenate((xmax, Const.grid_height - ymax, zeros), axis=1).reshape(1, n_bbox, 3)
    pth0s = np.concatenate((xmax, Const.grid_height - ymin, heights), axis=1).reshape(1, n_bbox, 3)
    pth1s = np.concatenate((xmin, Const.grid_height - ymin, heights), axis=1).reshape(1, n_bbox, 3)
    pth2s = np.concatenate((xmin, Const.grid_height - ymax, heights), axis=1).reshape(1, n_bbox, 3)
    pth3s = np.concatenate((xmax, Const.grid_height - ymax, heights), axis=1).reshape(1, n_bbox, 3)

    res = np.vstack((pt0s, pt1s, pt2s, pt3s, pth0s, pth1s, pth2s, pth3s)).transpose(1, 0, 2)
    return res

def getimage_pt(points3d, extrin, intrin):
    # 此处输入的是以左下角为原点的坐标，输出的是opencv格式的左上角为原点的坐标
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(extrin, newpoints3d)[-1]
    imagepoints = (np.dot(intrin, np.dot(extrin, newpoints3d)) / Zc).astype(np.int)
    # print(Zc)
    return [imagepoints[0, 0], imagepoints[1, 0]]

def getprojected_3dbox(points3ds, extrin, intrin, isleft = True):
    if isleft:
        extrin_ = extrin[0].numpy()
        intrin_ = intrin[0].numpy()
    else:
        extrin_ = extrin[1].numpy()
        intrin_ = intrin[1].numpy()
    # print(extrin_.shape)
    extrin_big = extrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)
    intrin_big = intrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)

    points3ds_big = points3ds.reshape(points3ds.shape[0], points3ds.shape[1], 3, 1)
    homog = np.ones((points3ds.shape[0], points3ds.shape[1], 1, 1))
    homo3dpts = np.concatenate((points3ds_big, homog), 2).reshape(points3ds.shape[0] * points3ds.shape[1], 4, 1)
    res = np.matmul(extrin_big, homo3dpts)
    Zc = res[:, -1]
    # print(intrin_big.shape, res.shape)
    res2 = np.matmul(intrin_big, res)
    imagepoints = (res2.reshape(-1, 3) / Zc).reshape((points3ds.shape[0], points3ds.shape[1], 3))[:, :, :2].astype(int)

    return imagepoints

def getprojected_3dbox_ori(points3ds, extrin, intrin, position_mark, isleft=True):
    # print("dzc", points3ds.shape, position_mark.shape)
    left_bboxes = []
    for i in range(points3ds.shape[0]):
        left_bbox_2d = []
        # print(points3ds[i].shape)
        for pt in points3ds[i]:
            # print(position_mark[i])
            if position_mark[i] == 0:
                left = getimage_pt(pt.reshape(3, 1), extrin[0][0], intrin[0][0])[:2]
            else:
                left = getimage_pt(pt.reshape(3, 1), extrin[1][0], intrin[1][0])[:2]
            left_bbox_2d.append(left)
        left_bboxes.append(left_bbox_2d)
        # print(left_bboxes)
    return np.array(left_bboxes).reshape((points3ds.shape[0], points3ds.shape[1], 2))

def getprojected_3dbox_right(points3ds, extrin, intrin):
    right_bboxes = []
    for i in range(points3ds.shape[0]):
        right_bbox_2d = []
        for pt in points3ds[i]:
            right = getimage_pt(pt.reshape(3, 1), extrin[1][0], intrin[1][0])
            right_bbox_2d.append(right)
        right_bboxes.append(right_bbox_2d)

    return np.array(right_bboxes).reshape((points3ds.shape[0], 9, 2))

def get_outter(projected_3dboxes):
    projected_3dboxes = projected_3dboxes + 1e-3
    zero_mask = np.zeros((projected_3dboxes.shape[0], projected_3dboxes.shape[1], 1))
    one_mask = np.ones((projected_3dboxes.shape[0], projected_3dboxes.shape[1], 1))
    huge_mask = one_mask * 1000
    ymax_mask = np.concatenate((zero_mask, one_mask), axis=2)
    xmax_mask = np.concatenate((one_mask, zero_mask), axis=2)
    ymin_mask = np.concatenate((huge_mask, one_mask), axis=2)
    xmin_mask = np.concatenate((one_mask, huge_mask), axis=2)
    xmax = np.max((projected_3dboxes * xmax_mask), axis = (1,2)).reshape(1, -1, 1)
    ymax = np.max((projected_3dboxes * ymax_mask), axis = (1,2)).reshape(1, -1, 1)
    xmin = np.min((projected_3dboxes * xmin_mask), axis = (1,2)).reshape(1, -1, 1)
    ymin = np.min((projected_3dboxes * ymin_mask), axis = (1,2)).reshape(1, -1, 1)
    res = np.concatenate((ymin, xmin, ymax, xmax), axis=2)
    res = np.array(res, dtype=int).squeeze()

    return res

def generate_3d_bbox2(pred_bboxs):
    # 输出以左下角为原点的3d坐标
    n_bbox = pred_bboxs.shape[0]
    boxes_3d = [] #
    for i in range(pred_bboxs.shape[0]):
        ymax, xmax, ymin, xmin = pred_bboxs[i]
        pt0 = [xmax, Const.grid_height - ymin, 0]
        pt1 = [xmin, Const.grid_height - ymin, 0]
        pt2 = [xmin, Const.grid_height - ymax, 0]
        pt3 = [xmax, Const.grid_height - ymax, 0]
        pt_h_0 = [xmax, Const.grid_height - ymin, Const.car_height]
        pt_h_1 = [xmin, Const.grid_height - ymin, Const.car_height]
        pt_h_2 = [xmin, Const.grid_height - ymax, Const.car_height]
        pt_h_3 = [xmax, Const.grid_height - ymax, Const.car_height]
        boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3])
    return np.array(boxes_3d).reshape((n_bbox, 8, 3))
