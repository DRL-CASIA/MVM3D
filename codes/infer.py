import math
import torch.nn.functional as F
import matplotlib
import sys
from codes.evaluation.pyeval.calAOS import evaluateDetectionAPAOS
from codes.evaluation.evaluate import evaluate
from codes.utils import array_tool as at
from torchvision.ops import boxes as box_ops
import time
import cv2
import os
from codes.EX_CONST import Const
import numpy as np
import torch
import torchvision.transforms as T
from codes.models.PPN import PerspTransDetector
from codes.utils.image_utils import img_color_denormalize
from codes.datasets.MVM3D import MVM3D
from codes.datasets.MVM3D_loader import MVM3D_loader
from codes.models.MbonHead import MbonHead
import argparse
os.environ['OMP_NUM_THREADS'] = '1'
matplotlib.use('Agg')
sys.path.append("../..")

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()

class Inferer(BaseTrainer):
    def __init__(self, model, roi_head, denormalize):
        self.model = model
        self.roi_head = roi_head
        self.denormalize = denormalize
        self.bins = Const.bins

    def infer(self, data_loader):
        all_pred_res = []
        all_gt_res = []

        for batch_idx, data in enumerate(data_loader):
            imgs, bev_xy, bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, left_dirs, right_dirs, left_sincos, right_sincos, left_orientation, right_orientation, left_conf, right_conf, frame, extrin, intrin, mark = data

            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, frame, mark=mark)
            roi = torch.tensor(rois)

            # -----------Projection------------
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

            left_2d_bbox = torch.tensor(left_2d_bbox)
            right_2d_bbox = torch.tensor(right_2d_bbox)

            # ------------MBON-----------
            left_roi_cls_loc, left_roi_score, left_pred_orientation, left_pred_conf = self.roi_head(
                img_featuremaps[0],
                torch.tensor(left_2d_bbox).to(img_featuremaps[0].device),
                left_rois_indices)

            right_roi_cls_loc, right_roi_score, right_pred_orientation, right_pred_conf = self.roi_head(
                img_featuremaps[1],
                torch.tensor(right_2d_bbox).to(img_featuremaps[1].device),
                right_rois_indices)

            angle_bins = generate_bins(self.bins)
            tmp = np.zeros(shape=left_pred_conf.shape)
            tmp += angle_bins
            left_angle_bins = tmp

            tmp = np.zeros(shape=right_pred_conf.shape)
            tmp += angle_bins
            right_angle_bins = tmp

            left_argmax = np.argmax(left_pred_conf.detach().cpu().numpy(), axis=1)
            left_orient = left_pred_orientation[np.arange(len(left_pred_orientation)), left_argmax]
            left_cos = left_orient[:, 0]
            left_sin = left_orient[:, 1]
            left_alpha = np.arctan2(left_sin.cpu().detach().numpy(), left_cos.cpu().detach().numpy())
            left_alpha += left_angle_bins[np.arange(len(left_argmax)), left_argmax]  # 0~180, (R, 2), residual angle
            right_argmax = np.argmax(right_pred_conf.detach().cpu().numpy(), axis=1)
            right_orient = right_pred_orientation[np.arange(len(right_pred_orientation)), right_argmax]
            right_cos = right_orient[:, 0]
            right_sin = right_orient[:, 1]
            right_alpha = np.arctan2(right_sin.cpu().detach().numpy(), right_cos.cpu().detach().numpy())
            right_alpha += right_angle_bins[np.arange(len(right_argmax)), right_argmax]  # 0~180, (R, 2), residual angle

            left_prob = at.tonumpy(F.softmax(at.totensor(left_roi_score), dim=1))
            left_front_prob = left_prob[:, 1]
            right_prob = at.tonumpy(F.softmax(at.totensor(right_roi_score), dim=1))
            right_front_prob = right_prob[:, 1]

            position_mark = np.concatenate(
                (np.zeros((left_front_prob.shape[0],)), np.ones((right_front_prob.shape[0]))))
            all_front_prob = np.concatenate((left_front_prob, right_front_prob))
            all_roi_remain = np.concatenate((roi[left_index_inside], roi[right_index_inside]))
            all_pred_alpha = np.concatenate((at.tonumpy(left_alpha), at.tonumpy(right_alpha)))

            v, indices = torch.tensor(all_front_prob).sort(0)
            indices_remain = indices[v > 0.6]
            print("Frame Number: ", frame.item())
            all_roi_remain = all_roi_remain[indices_remain].reshape(len(indices_remain), 4)
            all_pred_alpha = all_pred_alpha[indices_remain].reshape(len(indices_remain), 1)
            all_front_prob = all_front_prob[indices_remain].reshape(len(indices_remain), )
            position_mark = position_mark[indices_remain].reshape(len(indices_remain), 1)
            all_bev_boxes = []
            if indices_remain.shape[0] != 0:
                if indices_remain.shape[0] == 1:
                    keep = [0]
                else:
                    keep = box_ops.nms(torch.tensor(all_roi_remain), torch.tensor(all_front_prob), 0)
                all_bev_boxes, all_pred_alpha, position_mark_keep = all_roi_remain[keep].reshape(len(keep), 4), \
                                                                    all_pred_alpha[keep].reshape(len(keep), 1), \
                                                                    position_mark[keep].reshape(len(keep))

            if len(all_bev_boxes) != 0:
                test_gt_res, test_pred_res = visualize_3dbox(all_bev_boxes, all_pred_alpha, position_mark_keep, gt_bbox,
                                                             bev_angle, all_front_prob[keep], extrin, intrin, frame)
                for p in range(len(test_gt_res)):
                    all_gt_res.append(test_gt_res[p])
                for l in range(len(test_pred_res)):
                    all_pred_res.append(test_pred_res[l])

        res_fpath = '%s/res/all_res.txt' % Const.data_root
        all_gt_fpath = '%s/res/all_test_gt.txt' % Const.data_root
        all_gt_res = np.array(all_gt_res).reshape(-1, 14)
        all_pred_res = np.array(all_pred_res).reshape(-1, 15)

        np.savetxt(res_fpath, all_pred_res, "%f")
        np.savetxt(all_gt_fpath, all_gt_res, "%f")

        recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(all_gt_fpath),
                                                 data_loader.dataset.base.__name__)

        AP_50, AOS_50, OS_50, AP_25, AOS_25, OS_25 = evaluateDetectionAPAOS(res_fpath, all_gt_fpath)
        print()
        print("MODA: %.1f" % moda, ", MODP: %.1f" % modp, ", Prec .: %.1f" % precision, ", Recall: %.1f" % recall )
        print("AP_50: %.1f" % AP_50, " ,AOS_50: %.1f" % AOS_50, ", OS_50: %.2f" % OS_50)
        print("AP_25: %.1f" % AP_25, " ,AOS_25: %.1f" % AOS_25, ", OS_25: %.2f" % OS_25)

    @property
    def n_class(self):
        return self.roi_head.n_class

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of the bin

    return angle_bins


def visualize_3dbox(pred_ori, pred_alpha, position_mark, gt_bbox, bev_angle, all_front_prob, extrin, intrin, idx, isRoi=False):
    # left_img = cv2.imread("/home/dzc/Data/%s/img/left1/%d.jpg" % (Const.dataset, idx))
    right_img = cv2.imread("%s/img/right2/%d.jpg" % (Const.data_root, idx))
    all_pred_res = []
    all_gt_res = []

    n_bbox = pred_ori.shape[0]
    gt_bbox = gt_bbox[0]
    bev_angle = bev_angle[0]
    gt_n_bbox = gt_bbox.shape[0]
    # ======================Gt========================
    boxes_3d = []
    for j, bbox in enumerate(gt_bbox):
        ymin, xmin, ymax, xmax = bbox
        theta = bev_angle[j]
        center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
        w = Const.car_length
        h = Const.car_width

        xmin = center_x - w // 2
        xmax = center_x + w // 2
        ymin = center_y - h // 2
        ymax = center_y + h // 2

        x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 - 40
        y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height - ymin, Const.grid_height - ymax, Const.grid_height - ymax, Const.grid_height - ymin, (
                    Const.grid_height - ymax + Const.grid_height - ymin) / 2

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

        all_gt_res.append(
            [idx.item(), center_x, center_y, w, h, np.rad2deg(theta.item()), x1_rot, y1_rot, x2_rot, y2_rot, x3_rot,
             y3_rot, x4_rot, y4_rot])

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
    # gt_projected_2d = getprojected_3dbox(gt_ori, extrin, intrin, isleft=True)
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

        cv2.arrowedLine(right_img, (int((gt_projected_2d[k][0][0] + gt_projected_2d[k][2][0]) / 2),
                                    int((gt_projected_2d[k][0][1] + gt_projected_2d[k][2][1]) / 2)),
                        (gt_projected_2d[k][8][0], gt_projected_2d[k][8][1]), color=(255, 60, 199), thickness=2)
    cv2.imwrite("result_images/%d_gt.jpg" % (idx), right_img)



    right_img = cv2.imread("%s/img/right2/%d.jpg" % (Const.data_root, idx))

    boxes_3d = []
    for i, bbox in enumerate(pred_ori):
        ymin, xmin, ymax, xmax = bbox
        alpha = pred_alpha[i]
        if pred_alpha.shape[0] == 1:
            score = 1.0
        else:
            score = all_front_prob[i]
        if position_mark[i] == 0:
            center_x, center_y = int((bbox[1] + bbox[3]) // 2), int((bbox[0] + bbox[2]) // 2)
            w, h = Const.car_length, Const.car_width
            xmin = center_x - w//2
            xmax = center_x + w//2
            ymin = center_y - h//2
            ymax = center_y + h//2
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 + 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height - ymin, Const.grid_height - ymax, Const.grid_height - ymax, Const.grid_height - ymin, (
                        Const.grid_height - ymax + Const.grid_height - ymin) / 2
            center_x, center_y = int((bbox[1] + bbox[3]) // 2), int((bbox[0] + bbox[2]) // 2)
            ray = np.arctan((Const.grid_height - center_y) / center_x)
            angle = alpha
        else:
            center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
            w, h = Const.car_length, Const.car_width
            xmin = center_x - w // 2
            xmax = center_x + w // 2
            ymin = center_y - h // 2
            ymax = center_y + h // 2
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 - 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height - ymin, Const.grid_height - ymax, Const.grid_height - ymax, Const.grid_height - ymin, (
                        Const.grid_height - ymax + Const.grid_height - ymin) / 2
            center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
            ray = np.arctan(center_y / (Const.grid_width - center_x))
            angle = alpha
        theta_l = angle
        theta = theta_l + ray

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

        if position_mark[i] == 0:
            theta_left = theta + np.pi
            all_pred_res.append(
                [idx.item(), center_x, center_y, w, h, np.rad2deg(theta_left.item()), score, x1_rot, y1_rot, x2_rot,
                 y2_rot, x3_rot, y3_rot, x4_rot, y4_rot])
        else:
            all_pred_res.append(
                [idx.item(), center_x, center_y, w, h, np.rad2deg(theta.item()), score, x1_rot, y1_rot, x2_rot, y2_rot,
                 x3_rot, y3_rot, x4_rot, y4_rot])

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
    # projected_2d_left = getprojected_3dbox(pred_ori, extrin, intrin, isleft=True)
    projected_2d = getprojected_3dbox(pred_ori, extrin, intrin, isleft=False)
    for k in range(n_bbox):
        if position_mark[k] == 0:
            color = (255, 255, 0)
            color = (0, 255, 0)
        else:
            color = (100, 100, 200)
            color = (0, 255, 0)
        if position_mark[k] == 1 and isRoi:
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

        else:
            cv2.arrowedLine(right_img, (int((projected_2d[k][0][0] + projected_2d[k][2][0]) / 2),
                                        int((projected_2d[k][0][1] + projected_2d[k][2][1]) / 2)),
                            (projected_2d[k][8][0], projected_2d[k][8][1]), color=(255, 60, 199), thickness=2)
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
    else:
        cv2.imwrite("result_images/%d.jpg" % (idx), right_img)
    return all_gt_res, all_pred_res


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
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
    in_weight[(torch.tensor(gt_label) > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)

    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss


def generate_3d_bbox(pred_bboxs):
    # 输出以左下角为原点的3d坐标
    n_bbox = pred_bboxs.shape[0]
    zeros = np.zeros((n_bbox, 1))
    heights = np.zeros((n_bbox, 1)) * Const.car_height
    ymax, xmax, ymin, xmin = pred_bboxs[:, 0].reshape(-1, 1), pred_bboxs[:, 1].reshape(-1, 1), pred_bboxs[:, 2].reshape(
        -1, 1), pred_bboxs[:, 3].reshape(-1, 1)

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
    return [imagepoints[0, 0], imagepoints[1, 0]]


def getprojected_3dbox(points3ds, extrin, intrin, isleft=True):
    if isleft:
        extrin_ = extrin[0].numpy()
        intrin_ = intrin[0].numpy()
    else:
        extrin_ = extrin[1].numpy()
        intrin_ = intrin[1].numpy()
    extrin_big = extrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)
    intrin_big = intrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)

    points3ds_big = points3ds.reshape(points3ds.shape[0], points3ds.shape[1], 3, 1)
    homog = np.ones((points3ds.shape[0], points3ds.shape[1], 1, 1))
    homo3dpts = np.concatenate((points3ds_big, homog), 2).reshape(points3ds.shape[0] * points3ds.shape[1], 4, 1)
    res = np.matmul(extrin_big, homo3dpts)
    Zc = res[:, -1]
    res2 = np.matmul(intrin_big, res)
    imagepoints = (res2.reshape(-1, 3) / Zc).reshape((points3ds.shape[0], points3ds.shape[1], 3))[:, :, :2].astype(int)

    return imagepoints


def get_outter(projected_3dboxes):
    projected_3dboxes = projected_3dboxes + 1e-3
    zero_mask = np.zeros((projected_3dboxes.shape[0], projected_3dboxes.shape[1], 1))
    one_mask = np.ones((projected_3dboxes.shape[0], projected_3dboxes.shape[1], 1))
    huge_mask = one_mask * 1000
    ymax_mask = np.concatenate((zero_mask, one_mask), axis=2)
    xmax_mask = np.concatenate((one_mask, zero_mask), axis=2)
    ymin_mask = np.concatenate((huge_mask, one_mask), axis=2)
    xmin_mask = np.concatenate((one_mask, huge_mask), axis=2)
    xmax = np.max((projected_3dboxes * xmax_mask), axis=(1, 2)).reshape(1, -1, 1)
    ymax = np.max((projected_3dboxes * ymax_mask), axis=(1, 2)).reshape(1, -1, 1)
    xmin = np.min((projected_3dboxes * xmin_mask), axis=(1, 2)).reshape(1, -1, 1)
    ymin = np.min((projected_3dboxes * ymin_mask), axis=(1, 2)).reshape(1, -1, 1)
    res = np.concatenate((ymin, xmin, ymax, xmax), axis=2)
    res = np.array(res, dtype=int).squeeze()

    return res


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(7)
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_trans = T.Compose([T.ToTensor(), normalize])
    data_path = os.path.expanduser(Const.data_root)

    base = MVM3D(data_path, args, worldgrid_shape=Const.grid_size)

    test_set = MVM3D_loader(base, train=2, transform=test_trans, grid_reduce=Const.reduce)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=True)

    # model
    model = PerspTransDetector(test_set)
    mbon_head = MbonHead(Const.roi_classes + 1, 7, 1 / Const.reduce)
    trainer = Inferer(model, mbon_head, denormalize)
    model.load_state_dict(torch.load('pretrained_models/ppn.pth'))
    mbon_head.load_state_dict(torch.load('pretrained_models/mbon.pth'))

    mbon_head.eval()
    model.eval()
    trainer.infer(test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVM3D')
    parser.add_argument('-d', '--dataset', type=str, default='robo', choices=['wildtrack' ,'robo'])
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--seed', type=int, default=7, help='random seed (default: None)')
    parser.add_argument('--resume', type=bool, default = True)
    args = parser.parse_args()

    main(args)
