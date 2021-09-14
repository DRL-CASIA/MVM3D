import numpy as np
import torch

from .bbox_tools import bbox2loc, bbox_iou, loc2bbox
# from .nms import non_maximum_suppression
from codes.EX_CONST import Const
from torchvision.ops import boxes as box_ops

class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.35, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, gt_bev_bbox, left_label, right_label, left_angles, right_angles, left_gt_bbox, right_gt_bbox, extrin, intrin, frame,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        left_remove_idx = []
        right_remove_idx = []
        for i in range(len(left_gt_bbox)):
            if left_gt_bbox[i][0] == -1 and left_gt_bbox[i][1] == -1 and left_gt_bbox[i][2] == -1 and left_gt_bbox[i][3] == -1:
                left_remove_idx.append(i)
            if right_gt_bbox[i][0] == -1 and right_gt_bbox[i][1] == -1 and right_gt_bbox[i][2] == -1 and right_gt_bbox[i][3] == -1:
                right_remove_idx.append(i)
        # 得出左右两边需要删掉的那个框，再考虑剩下的事情

        gt_left_bev_bbox = np.delete(gt_bev_bbox, left_remove_idx, axis=0)
        gt_right_bev_bbox = np.delete(gt_bev_bbox, right_remove_idx, axis=0)
        left_gt_bbox = np.delete(left_gt_bbox, left_remove_idx, axis=0)
        right_gt_bbox = np.delete(right_gt_bbox, right_remove_idx, axis=0)

        # left
        # 限定用于左侧的roi
        roi_remain_idx = []
        for id, bbox in enumerate(roi):
            y = (bbox[0] + bbox[2]) / 2
            x = (bbox[1] + bbox[3]) / 2
            z = 0
            pt2d = getimage_pt(np.array([x, Const.grid_height - y, z]).reshape(3,1), extrin[0][0], intrin[0][0])
            if 0 < int(pt2d[0]) < 640 and 0 < int(pt2d[1]) < 480:
                roi_remain_idx.append(id)
        left_rois = roi[roi_remain_idx]

        # right_index_inside = np.where(
        #     (roi[:, 0] >= 0) &
        #     (roi[:, 1] >= 0) &
        #     (roi[:, 2] <= 640) &
        #     (roi[:, 3] <= Const.ori_img_width)
        # )[0]


        left_n_bbox, _ = gt_left_bev_bbox.shape
        left_roi = np.concatenate((left_rois, gt_left_bev_bbox), axis=0)
        left_pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        left_iou = bbox_iou(left_roi, gt_left_bev_bbox) # R， 4每个roi和gt的iou
        left_gt_assignment = left_iou.argmax(axis=1) # 每个roi对应iou最大的一个gt框的索引值
        left_max_iou = left_iou.max(axis=1) # 每个roi对应iou最大的一个gt框的置信度值
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        # print(left_angles.shape, left_gt_assignment.shape)
        gt_roi_angles_left = left_angles.reshape(-1, 2)[left_gt_assignment]
        gt_roi_label_left = left_label[left_gt_assignment] + 1 # 每一个roi对应的gt及其gt的分类


        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        left_pos_index = np.where(left_max_iou >= self.pos_iou_thresh)[0]
        left_pos_roi_per_this_image = int(min(left_pos_roi_per_image, left_pos_index.size))
        if left_pos_index.size > 0:
            left_pos_index = np.random.choice(
                left_pos_index, size=left_pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        left_neg_index = np.where((left_max_iou < self.neg_iou_thresh_hi) &
                             (left_max_iou >= self.neg_iou_thresh_lo))[0]
        left_neg_roi_per_this_image = self.n_sample - left_pos_roi_per_this_image
        left_neg_roi_per_this_image = int(min(left_neg_roi_per_this_image,
                                         left_neg_index.size))
        if left_neg_index.size > 0:
            left_neg_index = np.random.choice(
                left_neg_index, size=left_neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        left_keep_index = np.append(left_pos_index, left_neg_index) # 前left_pos_index个是参与角度回归的正样本
        left_gt_label = gt_roi_label_left[left_keep_index]
        left_gt_label[left_pos_roi_per_this_image:] = 0  # negative labels --> 0

        left_gt_angles = gt_roi_angles_left[left_pos_index] # only keep the positive samples for training
        left_sample_roi = left_roi[left_keep_index]

        # -------------------------------right--------------------------------------
        right_n_bbox, _ = gt_right_bev_bbox.shape
        right_roi = np.concatenate((roi, gt_right_bev_bbox), axis=0)
        right_pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        right_iou = bbox_iou(right_roi, gt_right_bev_bbox)  # R， 4每个roi和gt的iou
        right_gt_assignment = right_iou.argmax(axis=1)  # 每个roi对应iou最大的一个gt框的索引值
        right_max_iou = right_iou.max(axis=1)  # 每个roi对应iou最大的一个gt框的置信度值
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label_right = right_label[right_gt_assignment] + 1
        gt_roi_angles_right = right_angles.reshape(-1, 2)[right_gt_assignment]

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        right_pos_index = np.where(right_max_iou >= self.pos_iou_thresh)[0]
        right_pos_roi_per_this_image = int(min(right_pos_roi_per_image, right_pos_index.size))
        if right_pos_index.size > 0:
            right_pos_index = np.random.choice(
                right_pos_index, size=right_pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        right_neg_index = np.where((right_max_iou < self.neg_iou_thresh_hi) &
                             (right_max_iou >= self.neg_iou_thresh_lo))[0]
        right_neg_roi_per_this_image = self.n_sample - right_pos_roi_per_this_image
        right_neg_roi_per_this_image = int(min(right_neg_roi_per_this_image,
                                         right_neg_index.size))
        if right_neg_index.size > 0:
            right_neg_index = np.random.choice(
                right_neg_index, size=right_neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        right_keep_index = np.append(right_pos_index, right_neg_index)

        right_gt_label = gt_roi_label_right[right_keep_index]
        right_gt_label[right_pos_roi_per_this_image:] = 0  # negative labels --> 0

        right_gt_angles = gt_roi_angles_right[right_pos_index]

        right_sample_roi = right_roi[right_keep_index]
        # ------------------------开始转换坐标-------------------------
        left_roi_3d = generate_3d_bbox(left_sample_roi)
        left_2d_bbox = getprojected_3dbox(left_roi_3d, extrin[0][0], intrin[0][0])
        left_2d_bbox = get_outter(left_2d_bbox)

        right_roi_3d = generate_3d_bbox(right_sample_roi)
        right_2d_bbox = getprojected_3dbox(right_roi_3d, extrin[1][0], intrin[1][0])
        right_2d_bbox = get_outter(right_2d_bbox)

        # left_roi_3d = generate_3d_bbox2(left_sample_roi)
        # left_2d_bbox = getprojected_3dbox2(left_roi_3d, extrin, intrin, isleft=True)
        # left_2d_bbox = get_outter2(left_2d_bbox)
        #
        # right_roi_3d = generate_3d_bbox2(right_sample_roi)
        # right_2d_bbox = getprojected_3dbox2(right_roi_3d, extrin, intrin, isleft=False)
        # right_2d_bbox = get_outter2(right_2d_bbox)

        # print(left_2d_bbox2[:3], left_2d_bbox[:3])
        left_gt_roi_loc = bbox2loc(left_2d_bbox, left_gt_bbox[left_gt_assignment[left_keep_index]])

        # left_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/left1/%d.jpg" % frame)
        # for idx, bbx in enumerate(left_gt_bbox[left_gt_assignment[left_keep_index]]):
        #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=3)
        # for idx, bbx in enumerate(left_2d_bbox):
        #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(192, 0, 120), thickness=1)
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/left_img_%d.jpg" % frame, left_img)
        left_gt_loc = ((left_gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))
        # print(left_gt_loc, right_2d_bbox)
        # Compute offsets and scales to match sampled RoIs to the GTs.
        # print((gt_bev_bbox[gt_assignment[keep_index]]).shape)
        right_gt_roi_loc = bbox2loc(right_2d_bbox, right_gt_bbox[right_gt_assignment[right_keep_index]])
        right_gt_loc = ((right_gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return left_2d_bbox, left_sample_roi, left_gt_loc, left_gt_label, left_gt_angles, len(left_pos_index), right_2d_bbox, right_sample_roi, right_gt_loc, right_gt_label, right_gt_angles, len(right_pos_index)

class ProposalTargetCreator_conf(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=64,
                 pos_ratio=0.35, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.1, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, gt_bev_bbox, left_label, right_label, left_angles, right_angles, left_orientation, right_orientation, left_conf, right_conf, left_gt_bbox, right_gt_bbox, extrin, intrin, frame,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        left_remove_idx = []
        right_remove_idx = []
        for i in range(len(left_gt_bbox)):
            if left_gt_bbox[i][0] == -1 and left_gt_bbox[i][1] == -1 and left_gt_bbox[i][2] == -1 and left_gt_bbox[i][3] == -1:
                left_remove_idx.append(i)
            if right_gt_bbox[i][0] == -1 and right_gt_bbox[i][1] == -1 and right_gt_bbox[i][2] == -1 and right_gt_bbox[i][3] == -1:
                right_remove_idx.append(i)
        # 得出左右两边需要删掉的那个框，再考虑剩下的事情

        gt_left_bev_bbox = np.delete(gt_bev_bbox, left_remove_idx, axis=0)
        gt_right_bev_bbox = np.delete(gt_bev_bbox, right_remove_idx, axis=0)
        left_gt_bbox = np.delete(left_gt_bbox, left_remove_idx, axis=0)
        right_gt_bbox = np.delete(right_gt_bbox, right_remove_idx, axis=0)
        # left_orientatin = np.delete(left_orientation, left_remove_idx, axis=0)
        # right_orientatin = np.delete(right_orientation, right_remove_idx, axis=0)
        # left_conf = np.delete(left_conf, left_remove_idx, axis=0)
        # right_conf = np.delete(right_conf, right_remove_idx, axis=0)

        # left
        # 限定用于左侧的roi
        roi_remain_idx = []
        for id, bbox in enumerate(roi):
            y = (bbox[0] + bbox[2]) / 2
            x = (bbox[1] + bbox[3]) / 2
            z = 0
            pt2d = getimage_pt(np.array([x, Const.grid_height - y, z]).reshape(3,1), extrin[0][0], intrin[0][0])
            if 0 < int(pt2d[0]) < 640 and 0 < int(pt2d[1]) < 480:
                roi_remain_idx.append(id)
        left_rois = roi[roi_remain_idx]

        # right_index_inside = np.where(
        #     (roi[:, 0] >= 0) &
        #     (roi[:, 1] >= 0) &
        #     (roi[:, 2] <= 640) &
        #     (roi[:, 3] <= Const.ori_img_width)
        # )[0]


        left_n_bbox, _ = gt_left_bev_bbox.shape
        left_roi = np.concatenate((left_rois, gt_left_bev_bbox), axis=0)
        left_pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        left_iou = bbox_iou(left_roi, gt_left_bev_bbox) # R， 4每个roi和gt的iou
        left_gt_assignment = left_iou.argmax(axis=1) # 每个roi对应iou最大的一个gt框的索引值
        left_max_iou = left_iou.max(axis=1) # 每个roi对应iou最大的一个gt框的置信度值
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        # print(left_angles.shape, left_gt_assignment.shape)
        gt_roi_angles_left = left_angles.reshape(-1, 2)[left_gt_assignment]
        gt_roi_label_left = left_label[left_gt_assignment] + 1 # 每一个roi对应的gt及其gt的分类

        gt_roi_orientations_left = left_orientation.reshape(-1, Const.bins, 2)[left_gt_assignment]
        gt_roi_conf_left = left_conf.reshape(-1, Const.bins)[left_gt_assignment]

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        left_pos_index = np.where(left_max_iou >= self.pos_iou_thresh)[0]
        left_pos_roi_per_this_image = int(min(left_pos_roi_per_image, left_pos_index.size))
        if left_pos_index.size > 0:
            left_pos_index = np.random.choice(
                left_pos_index, size=left_pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        left_neg_index = np.where((left_max_iou < self.neg_iou_thresh_hi) &
                             (left_max_iou >= self.neg_iou_thresh_lo))[0]
        left_neg_roi_per_this_image = self.n_sample - left_pos_roi_per_this_image
        left_neg_roi_per_this_image = int(min(left_neg_roi_per_this_image,
                                         left_neg_index.size))
        if left_neg_index.size > 0:
            left_neg_index = np.random.choice(
                left_neg_index, size=left_neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        left_keep_index = np.append(left_pos_index, left_neg_index) # 前left_pos_index个是参与角度回归的正样本
        left_gt_label = gt_roi_label_left[left_keep_index]
        left_gt_label[left_pos_roi_per_this_image:] = 0  # negative labels --> 0

        left_gt_angles = gt_roi_angles_left[left_pos_index] # only keep the positive samples for training
        left_gt_orientation = gt_roi_orientations_left[left_pos_index]
        left_gt_conf = gt_roi_conf_left[left_pos_index]

        left_sample_roi = left_roi[left_keep_index]

        # -------------------------------right--------------------------------------
        right_n_bbox, _ = gt_right_bev_bbox.shape
        right_roi = np.concatenate((roi, gt_right_bev_bbox), axis=0)
        right_pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        right_iou = bbox_iou(right_roi, gt_right_bev_bbox)  # R， 4每个roi和gt的iou
        right_gt_assignment = right_iou.argmax(axis=1)  # 每个roi对应iou最大的一个gt框的索引值
        right_max_iou = right_iou.max(axis=1)  # 每个roi对应iou最大的一个gt框的置信度值
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label_right = right_label[right_gt_assignment] + 1
        gt_roi_angles_right = right_angles.reshape(-1, 2)[right_gt_assignment]

        gt_roi_orientations_right = right_orientation.reshape(-1, Const.bins, 2)[right_gt_assignment]
        gt_roi_conf_right = right_conf.reshape(-1, Const.bins)[right_gt_assignment]

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        right_pos_index = np.where(right_max_iou >= self.pos_iou_thresh)[0]
        right_pos_roi_per_this_image = int(min(right_pos_roi_per_image, right_pos_index.size))
        if right_pos_index.size > 0:
            right_pos_index = np.random.choice(
                right_pos_index, size=right_pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        right_neg_index = np.where((right_max_iou < self.neg_iou_thresh_hi) &
                             (right_max_iou >= self.neg_iou_thresh_lo))[0]
        right_neg_roi_per_this_image = self.n_sample - right_pos_roi_per_this_image
        right_neg_roi_per_this_image = int(min(right_neg_roi_per_this_image,
                                         right_neg_index.size))
        if right_neg_index.size > 0:
            right_neg_index = np.random.choice(
                right_neg_index, size=right_neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        right_keep_index = np.append(right_pos_index, right_neg_index)

        right_gt_label = gt_roi_label_right[right_keep_index]
        right_gt_label[right_pos_roi_per_this_image:] = 0  # negative labels --> 0

        right_gt_angles = gt_roi_angles_right[right_pos_index]
        right_gt_orientations = gt_roi_orientations_right[right_pos_index]
        right_gt_conf = gt_roi_conf_right[right_pos_index]

        right_sample_roi = right_roi[right_keep_index]
        # ------------------------开始转换坐标-------------------------
        left_roi_3d = generate_3d_bbox(left_sample_roi)
        left_2d_bbox = getprojected_3dbox(left_roi_3d, extrin[0][0], intrin[0][0])
        left_2d_bbox = get_outter(left_2d_bbox)

        right_roi_3d = generate_3d_bbox(right_sample_roi)
        right_2d_bbox = getprojected_3dbox(right_roi_3d, extrin[1][0], intrin[1][0])
        right_2d_bbox = get_outter(right_2d_bbox)

        # left_roi_3d = generate_3d_bbox2(left_sample_roi)
        # left_2d_bbox = getprojected_3dbox2(left_roi_3d, extrin, intrin, isleft=True)
        # left_2d_bbox = get_outter2(left_2d_bbox)
        #
        # right_roi_3d = generate_3d_bbox2(right_sample_roi)
        # right_2d_bbox = getprojected_3dbox2(right_roi_3d, extrin, intrin, isleft=False)
        # right_2d_bbox = get_outter2(right_2d_bbox)

        # print(left_2d_bbox2[:3], left_2d_bbox[:3])
        left_gt_roi_loc = bbox2loc(left_2d_bbox, left_gt_bbox[left_gt_assignment[left_keep_index]])

        # left_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/left1/%d.jpg" % frame)
        # for idx, bbx in enumerate(left_gt_bbox[left_gt_assignment[left_keep_index]]):
        #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=3)
        # for idx, bbx in enumerate(left_2d_bbox):
        #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(192, 0, 120), thickness=1)
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/left_img_%d.jpg" % frame, left_img)
        left_gt_loc = ((left_gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))
        # print(left_gt_loc, right_2d_bbox)
        # Compute offsets and scales to match sampled RoIs to the GTs.
        # print((gt_bev_bbox[gt_assignment[keep_index]]).shape)
        right_gt_roi_loc = bbox2loc(right_2d_bbox, right_gt_bbox[right_gt_assignment[right_keep_index]])
        right_gt_loc = ((right_gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return left_2d_bbox, left_sample_roi, left_gt_loc, left_gt_label, left_gt_angles, len(left_pos_index), right_2d_bbox, right_sample_roi, right_gt_loc, right_gt_label, right_gt_angles, len(right_pos_index), left_gt_orientation, left_gt_conf, right_gt_orientations, right_gt_conf

class ProposalTargetCreator_ori(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.6,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        # print(sample_roi.shape, bbox[gt_assignment[keep_index]].shape)
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])

        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))
        return sample_roi, gt_roi_loc, gt_roi_label

class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size
        n_anchor = len(anchor)

        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]

        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        # -----------------------------------------------------------
        # tmp = np.zeros((Const.grid_height, Const.grid_width), dtype=np.uint8())
        # import cv2
        # tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        #
        # for idx, anc in enumerate(anchor):
        #     if label[inside_index][idx] == 1:
        #         cv2.rectangle(tmp, (int(anc[1]), int(anc[0])), (int(anc[3]), int(anc[2])), color=(255, 255, 0))
        #
        # for idx, bbx in enumerate(bbox):
        #     cv2.rectangle(tmp, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(34, 34,178), thickness=2)
        #
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/anchorBase.jpg", tmp)
        # # -----------------------------------------------------------
        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious

def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=3000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=50,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        # print(img_size)
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).

        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        roi_origin = roi.copy()
        score = score[order]

        # s = time.time()
        keep = box_ops.nms(torch.tensor(roi), torch.tensor(score.copy(), dtype = torch.tensor(roi).dtype), self.nms_thresh).numpy() # torchvision nms
        # e = time.time()
        # print(e - s)

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu

        # keep2 = non_maximum_suppression(
        #     cp.ascontiguousarray(cp.asarray(roi)),
        #     thresh=self.nms_thresh)

        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi, roi_origin, order.copy()

def generate_3d_bbox2(pred_bboxs):
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

def getimage_pt2(points3d, extrin, intrin):
    # 此处输入的是以左下角为原点的坐标，输出的是opencv格式的左上角为原点的坐标
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(extrin, newpoints3d)[-1]
    imagepoints = (np.dot(intrin, np.dot(extrin, newpoints3d)) / Zc).astype(np.int)
    return [imagepoints[0, 0], imagepoints[1, 0]]

def getprojected_3dbox2(points3ds, extrin, intrin, isleft = True):
    if isleft:
        extrin_ = extrin[0].numpy()
        intrin_ = intrin[0].numpy()
    else:
        extrin_ = extrin[1].numpy()
        intrin_ = intrin[1].numpy()

    extrin_big = extrin_.repeat(points3ds.shape[0] * 8, axis=0)
    intrin_big = intrin_.repeat(points3ds.shape[0] * 8, axis=0)

    points3ds_big = points3ds.reshape(points3ds.shape[0], 8, 3, 1)
    homog = np.ones((points3ds.shape[0], 8, 1, 1))
    homo3dpts = np.concatenate((points3ds_big, homog), 2).reshape(points3ds.shape[0] * 8, 4, 1)
    res = np.matmul(extrin_big, homo3dpts)
    Zc = res[:, -1]
    res2 = np.matmul(intrin_big, res)
    imagepoints = (res2.reshape(-1, 3) / Zc).reshape((points3ds.shape[0], 8, 3))[:, :, :2].astype(int)

    return imagepoints

def get_outter2(projected_3dboxes):
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
    res = np.array(res).squeeze()

    return res


def generate_3d_bbox(pred_bboxs):
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

def getimage_pt(points3d, extrin, intrin):
    # 此处输入的是以左下角为原点的坐标，输出的是opencv格式的左上角为原点的坐标
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(extrin, newpoints3d)[-1]
    imagepoints = (np.dot(intrin, np.dot(extrin, newpoints3d)) / Zc).astype(np.int)
    return [imagepoints[0, 0], imagepoints[1, 0]]

def getprojected_3dbox(points3ds, extrin, intrin):
    bboxes = []
    for i in range(points3ds.shape[0]):
        bbox_2d = []
        for pt in points3ds[i]:
            left = getimage_pt(pt.reshape(3, 1), extrin, intrin)
            bbox_2d.append(left)
        bboxes.append(bbox_2d)

    return np.array(bboxes).reshape((points3ds.shape[0], 8, 2))

def get_outter(projected_3dboxes):
    outter_boxes = []
    for boxes in projected_3dboxes:
        xmax = max(boxes[:, 0])
        xmin = min(boxes[:, 0])
        ymax = max(boxes[:, 1])
        ymin = min(boxes[:, 1])
        outter_boxes.append([ymin, xmin, ymax, xmax])
    return np.array(outter_boxes, dtype=np.float)